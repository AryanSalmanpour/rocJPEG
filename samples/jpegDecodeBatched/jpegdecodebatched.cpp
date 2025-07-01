/*
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "../rocjpeg_samples_utils.h"

struct BatchJob {
    std::vector<std::vector<char>> batch_images;
    std::vector<RocJpegStreamHandle> stream_handles;
    std::vector<RocJpegImage> output_images;
    std::vector<RocJpegDecodeParams> decode_params_batch;
    std::vector<std::vector<uint32_t>> widths;
    std::vector<std::vector<uint32_t>> heights;
    std::vector<RocJpegChromaSubsampling> subsamplings;
    std::vector<std::string> base_file_names;
    std::vector<std::vector<uint32_t>> prior_channel_sizes;
    int current_batch_size = 0;
};

class BatchQueue {
public:
    BatchQueue(size_t max_size) : max_size_(max_size), shutdown_flag_(false) {}

    void Push(BatchJob&& job) {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [this] { return queue_.size() < max_size_ || shutdown_flag_; });
        if (shutdown_flag_) return;
        queue_.push(std::move(job));
        cv_.notify_all();
    }

    // Returns false if shutdown was signaled and queue is empty
    bool Front(BatchJob& job) {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [this] { return !queue_.empty() || shutdown_flag_; });
        if (queue_.empty() && shutdown_flag_) return false;
        job = queue_.front();
        return true;
    }

    void Pop() {
        std::unique_lock<std::mutex> lock(mtx_);
        if (!queue_.empty()) {
            queue_.pop();
            cv_.notify_all();
        }
    }

    void ShutDown() {
        {
            std::lock_guard<std::mutex> lock(mtx_);
            shutdown_flag_ = true;
        }
        cv_.notify_all();
    }

private:
    std::queue<BatchJob> queue_;
    std::mutex mtx_;
    std::condition_variable cv_;
    size_t max_size_;
    std::atomic<bool> shutdown_flag_;
};

constexpr int max_queue_size = 4;
BatchJob batch_jobs[max_queue_size];
std::mutex mtx_;
std::condition_variable cv_;
std::queue<int> queue_;
std::atomic<bool> stop_decoder_thread_(false);

void DecoderThreadFunc(RocJpegHandle rocjpeg_handle, RocJpegUtils &rocjpeg_utils, int &total_images, bool save_images, std::string &output_file_path, bool is_dir, double &time_per_image_all, double &mpixels_all) {
    int current_batch_index;
    BatchJob current_batch_job;
    uint32_t roi_width;
    uint32_t roi_height;
    bool is_roi_valid = false;
    while (true) {
        {
            std::unique_lock<std::mutex> lock(mtx_);
            cv_.wait(lock, [&]() {return !queue_.empty() || stop_decoder_thread_.load(); });
            if (stop_decoder_thread_.load() && queue_.empty()) {
                break;
            }
            current_batch_index = queue_.front();
            current_batch_job = batch_jobs[current_batch_index];
        }

        double time_per_batch_in_milli_sec = 0;
        if (current_batch_job.current_batch_size > 0) {
            auto start_time = std::chrono::high_resolution_clock::now();
            CHECK_ROCJPEG(rocJpegDecodeBatched(rocjpeg_handle, current_batch_job.stream_handles.data(), current_batch_job.current_batch_size, current_batch_job.decode_params_batch.data(), current_batch_job.output_images.data()));
            auto end_time = std::chrono::high_resolution_clock::now();
            time_per_batch_in_milli_sec = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        }

        double image_size_in_mpixels = 0;
        for (int b = 0; b < current_batch_job.current_batch_size; b++) {
            image_size_in_mpixels += (static_cast<double>(current_batch_job.widths[b][0]) * static_cast<double>(current_batch_job.heights[b][0]) / 1000000);
        }

        total_images += current_batch_job.current_batch_size;

        if (save_images) {
            for (int b = 0; b < current_batch_job.current_batch_size; b++) {
                std::string image_save_path = output_file_path;
                //if ROI is present, need to pass roi_width and roi_height
                roi_width = current_batch_job.decode_params_batch[b].crop_rectangle.right - current_batch_job.decode_params_batch[b].crop_rectangle.left;
                roi_height = current_batch_job.decode_params_batch[b].crop_rectangle.bottom - current_batch_job.decode_params_batch[b].crop_rectangle.top;
                is_roi_valid = (roi_width > 0 && roi_height > 0 && roi_width <= current_batch_job.widths[b][0] && roi_height <= current_batch_job.heights[b][0]) ? true : false;
                uint32_t width = is_roi_valid ? roi_width : current_batch_job.widths[b][0];
                uint32_t height = is_roi_valid ? roi_height : current_batch_job.heights[b][0];
                if (is_dir) {
                    rocjpeg_utils.GetOutputFileExt(current_batch_job.decode_params_batch[b].output_format, current_batch_job.base_file_names[b], width, height, current_batch_job.subsamplings[b], image_save_path);
                }
                rocjpeg_utils.SaveImage(image_save_path, &current_batch_job.output_images[b], width, height, current_batch_job.subsamplings[b], current_batch_job.decode_params_batch[b].output_format);
            }
        }

        time_per_image_all += time_per_batch_in_milli_sec;
        mpixels_all += image_size_in_mpixels;

        {
            std::unique_lock<std::mutex> lock(mtx_);
            queue_.pop();
        }
        cv_.notify_one();
    }
}

int main(int argc, char **argv) {
    int device_id = 0;
    bool save_images = false;
    uint8_t num_components;
    uint32_t channel_sizes[ROCJPEG_MAX_COMPONENT] = {};
    uint32_t num_channels = 0;
    int total_images = 0;
    int batch_size = 2;
    double time_per_image_all = 0;
    double mpixels_all = 0;
    double images_per_sec = 0;
    std::string chroma_sub_sampling = "";
    std::string input_path, output_file_path;
    std::vector<std::string> file_paths = {};
    bool is_dir = false;
    bool is_file = false;
    RocJpegBackend rocjpeg_backend = ROCJPEG_BACKEND_HARDWARE;
    RocJpegHandle rocjpeg_handle = nullptr;
    RocJpegHandle rocjpeg_handle_thread = nullptr;
    std::vector<std::vector<RocJpegStreamHandle>> rocjpeg_stream_handles;
    RocJpegDecodeParams decode_params = {};
    RocJpegUtils rocjpeg_utils;
    std::vector<uint32_t> temp_widths(ROCJPEG_MAX_COMPONENT, 0);
    std::vector<uint32_t> temp_heights(ROCJPEG_MAX_COMPONENT, 0);
    RocJpegChromaSubsampling temp_subsampling;
    std::string temp_base_file_name;
    uint64_t num_bad_jpegs = 0;
    uint64_t num_jpegs_with_411_subsampling = 0;
    uint64_t num_jpegs_with_unknown_subsampling = 0;
    uint64_t num_jpegs_with_unsupported_resolution = 0;
    int current_batch_size = 0;
    int batch_index = 0;

    RocJpegUtils::ParseCommandLine(input_path, output_file_path, save_images, device_id, rocjpeg_backend, decode_params, nullptr, &batch_size, argc, argv);
    if (!RocJpegUtils::GetFilePaths(input_path, file_paths, is_dir, is_file)) {
        std::cerr << "ERROR: Failed to get input file paths!" << std::endl;
        return EXIT_FAILURE;
    }
    if (!RocJpegUtils::InitHipDevice(device_id)) {
        std::cerr << "ERROR: Failed to initialize HIP!" << std::endl;
        return EXIT_FAILURE;
    }

    CHECK_ROCJPEG(rocJpegCreate(rocjpeg_backend, device_id, &rocjpeg_handle));
    CHECK_ROCJPEG(rocJpegCreate(rocjpeg_backend, device_id, &rocjpeg_handle_thread));

    batch_size = std::min(batch_size, static_cast<int>(file_paths.size()));

    std::thread decoder_thread(DecoderThreadFunc, rocjpeg_handle_thread, std::ref(rocjpeg_utils), std::ref(total_images), save_images, std::ref(output_file_path), is_dir, std::ref(time_per_image_all), std::ref(mpixels_all));

    BatchQueue batch_queue(max_queue_size);
    rocjpeg_stream_handles.resize(max_queue_size);
    // create stream handles of batch size
    for (auto i = 0; i < max_queue_size; i++) {
        rocjpeg_stream_handles[i].resize(batch_size);
    }
    for (auto i = 0; i < max_queue_size; i++) {
        for (auto j = 0; j < batch_size; j++) {
            CHECK_ROCJPEG(rocJpegStreamCreate(&rocjpeg_stream_handles[i][j]));
        }
    }
    for (int i = 0; i < max_queue_size; i++) {
        batch_jobs[i].batch_images.resize(batch_size);
        batch_jobs[i].output_images.resize(batch_size);
        batch_jobs[i].decode_params_batch.resize(batch_size, decode_params);
        batch_jobs[i].prior_channel_sizes.resize(batch_size, std::vector<uint32_t>(ROCJPEG_MAX_COMPONENT, 0));
        batch_jobs[i].widths.resize(batch_size, std::vector<uint32_t>(ROCJPEG_MAX_COMPONENT, 0));
        batch_jobs[i].heights.resize(batch_size, std::vector<uint32_t>(ROCJPEG_MAX_COMPONENT, 0));
        batch_jobs[i].subsamplings.resize(batch_size);
        batch_jobs[i].base_file_names.resize(batch_size);
        batch_jobs[i].stream_handles.resize(batch_size);
    }
    std::cout << "Decoding started, please wait! ... " << std::endl;
    for (int i = 0; i < file_paths.size(); i += batch_size) {
        int batch_end = std::min(i + batch_size, static_cast<int>(file_paths.size()));
        for (int j = i; j < batch_end; j++) {
            int index = j - i;
            temp_base_file_name = file_paths[j].substr(file_paths[j].find_last_of("/\\") + 1);
            // Read an image from disk.
            std::ifstream input(file_paths[j].c_str(), std::ios::in | std::ios::binary | std::ios::ate);
            if (!(input.is_open())) {
                std::cerr << "ERROR: Cannot open image: " << file_paths[j] << std::endl;
                return EXIT_FAILURE;
            }
            // Get the size
            std::streamsize file_size = input.tellg();
            input.seekg(0, std::ios::beg);
            // resize if buffer is too small
            if (batch_jobs[batch_index].batch_images[index].size() < file_size) {
                batch_jobs[batch_index].batch_images[index].resize(file_size);
            }
            if (!input.read(batch_jobs[batch_index].batch_images[index].data(), file_size)) {
                std::cerr << "ERROR: Cannot read from file: " << file_paths[j] << std::endl;
                return EXIT_FAILURE;
            }

            RocJpegStatus rocjpeg_status = rocJpegStreamParse(reinterpret_cast<uint8_t*>(batch_jobs[batch_index].batch_images[index].data()), file_size, rocjpeg_stream_handles[batch_index][index]);
            if (rocjpeg_status != ROCJPEG_STATUS_SUCCESS) {
                if (is_dir) {
                    num_bad_jpegs++;
                    std::cerr << "Skipping decoding input file: " << file_paths[j] << std::endl;
                    continue;
                } else {
                    std::cerr << "ERROR: Failed to parse the input jpeg stream with " << rocJpegGetErrorName(rocjpeg_status) << std::endl;
                    return EXIT_FAILURE;
                }
            }

            CHECK_ROCJPEG(rocJpegGetImageInfo(rocjpeg_handle, rocjpeg_stream_handles[batch_index][index], &num_components, &temp_subsampling, temp_widths.data(), temp_heights.data()));

            rocjpeg_utils.GetChromaSubsamplingStr(temp_subsampling, chroma_sub_sampling);
            if (temp_widths[0] < 64 || temp_heights[0] < 64) {
                if (is_dir) {
                    num_jpegs_with_unsupported_resolution++;
                    continue;
                } else {
                    std::cerr << "The image resolution is not supported by VCN Hardware" << std::endl;
                    return EXIT_FAILURE;
                }
            }

            if (temp_subsampling == ROCJPEG_CSS_411 || temp_subsampling == ROCJPEG_CSS_UNKNOWN) {
                if (is_dir) {
                    if (temp_subsampling == ROCJPEG_CSS_411) {
                        num_jpegs_with_411_subsampling++;
                    }
                    if (temp_subsampling == ROCJPEG_CSS_UNKNOWN) {
                        num_jpegs_with_unknown_subsampling++;
                    }
                    continue;
                } else {
                    std::cerr << "The chroma sub-sampling is not supported by VCN Hardware" << std::endl;
                    return EXIT_FAILURE;
                }
            }

            if (rocjpeg_utils.GetChannelPitchAndSizes(batch_jobs[batch_index].decode_params_batch[index], temp_subsampling, temp_widths.data(), temp_heights.data(), num_channels, batch_jobs[batch_index].output_images[current_batch_size], channel_sizes)) {
                std::cerr << "ERROR: Failed to get the channel pitch and sizes" << std::endl;
                return EXIT_FAILURE;
            }

            // allocate memory for each channel and reuse them if the sizes remain unchanged for a new image.
            for (int n = 0; n < num_channels; n++) {
                if (batch_jobs[batch_index].prior_channel_sizes[current_batch_size][n] != channel_sizes[n]) {
                    if (batch_jobs[batch_index].output_images[current_batch_size].channel[n] != nullptr) {
                        CHECK_HIP(hipFree((void *)batch_jobs[batch_index].output_images[current_batch_size].channel[n]));
                        batch_jobs[batch_index].output_images[current_batch_size].channel[n] = nullptr;
                    }
                    CHECK_HIP(hipMalloc(&batch_jobs[batch_index].output_images[current_batch_size].channel[n], channel_sizes[n]));
                    batch_jobs[batch_index].prior_channel_sizes[current_batch_size][n] = channel_sizes[n];
                }
            }
            batch_jobs[batch_index].stream_handles[current_batch_size] = rocjpeg_stream_handles[batch_index][index];
            batch_jobs[batch_index].subsamplings[current_batch_size] = temp_subsampling;
            batch_jobs[batch_index].widths[current_batch_size] = temp_widths;
            batch_jobs[batch_index].heights[current_batch_size] = temp_heights;
            batch_jobs[batch_index].base_file_names[current_batch_size] = temp_base_file_name;
            current_batch_size++;
        }
        batch_jobs[batch_index].current_batch_size = current_batch_size;
        {
            std::unique_lock<std::mutex> lock(mtx_);
            cv_.wait(lock, [&] {return queue_.size() < max_queue_size;});
            queue_.push(batch_index);
        }
        batch_index = (batch_index + 1) % max_queue_size;
        cv_.notify_one();
        current_batch_size = 0;
    }

    {
        std::lock_guard<std::mutex> lock(mtx_);
        stop_decoder_thread_ = true;
    }
    cv_.notify_one();
    decoder_thread.join();

    if (is_dir) {
        time_per_image_all = time_per_image_all / total_images;
        images_per_sec = 1000 / time_per_image_all;
        double mpixels_per_sec = mpixels_all * images_per_sec / total_images;
        std::cout << "Total decoded images: " << total_images << std::endl;
        if (num_bad_jpegs || num_jpegs_with_411_subsampling || num_jpegs_with_unknown_subsampling || num_jpegs_with_unsupported_resolution) {
            std::cout << "Total skipped images: " << num_bad_jpegs + num_jpegs_with_411_subsampling + num_jpegs_with_unknown_subsampling + num_jpegs_with_unsupported_resolution;
            if (num_bad_jpegs) {
                std::cout << " ,total images that cannot be parsed: " << num_bad_jpegs;
            }
            if (num_jpegs_with_411_subsampling) {
                std::cout << " ,total images with YUV 4:1:1 chroam subsampling: " << num_jpegs_with_411_subsampling;
            }
            if (num_jpegs_with_unknown_subsampling) {
                std::cout << " ,total images with unknwon chroam subsampling: " << num_jpegs_with_unknown_subsampling;
            }
            if (num_jpegs_with_unsupported_resolution) {
                std::cout << " ,total images with unsupported_resolution: " << num_jpegs_with_unsupported_resolution;
            }
            std::cout << std::endl;
        }
        if (total_images) {
            std::cout << "Average processing time per image (ms): " << time_per_image_all << std::endl;
            std::cout << "Average decoded images per sec (Images/Sec): " << images_per_sec << std::endl;
            std::cout << "Average decoded images size (Mpixels/Sec): " << mpixels_per_sec << std::endl;
        }
    }

    //cleanup
    for (int n = 0; n < max_queue_size; n++) {
        for (auto& it : batch_jobs[n].output_images) {
            for (int i = 0; i < ROCJPEG_MAX_COMPONENT; i++) {
                if (it.channel[i] != nullptr) {
                    CHECK_HIP(hipFree((void *)it.channel[i]));
                    it.channel[i] = nullptr;
                }
            }
        }
    }
    CHECK_ROCJPEG(rocJpegDestroy(rocjpeg_handle));
    CHECK_ROCJPEG(rocJpegDestroy(rocjpeg_handle_thread));
    for (int n = 0; n < max_queue_size; n++) {
        for(auto& it : rocjpeg_stream_handles[n]) {
            CHECK_ROCJPEG(rocJpegStreamDestroy(it));
        }
    }

    std::cout << "Decoding completed!" << std::endl;
    return EXIT_SUCCESS;
}