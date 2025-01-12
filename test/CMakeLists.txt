# ##############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# ##############################################################################
cmake_minimum_required(VERSION 3.10)

if(NOT DEFINED ENHANCED_MESSAGE)
  string(ASCII 27 Esc)
  set(ColourReset "${Esc}[m")
  set(Red         "${Esc}[31m")
  set(Green       "${Esc}[32m")
  set(Yellow      "${Esc}[33m")
  set(White       "${Esc}[37m")
endif()

# ROCM Path
if(DEFINED ENV{ROCM_PATH})
  set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "${White}${PROJECT_NAME}: Default ROCm installation path${ColourReset}")
elseif(ROCM_PATH)
  message("-- ${White}${PROJECT_NAME} :ROCM_PATH Set -- ${ROCM_PATH}${ColourReset}")
else()
  set(ROCM_PATH /opt/rocm CACHE PATH "${White}${PROJECT_NAME}: Default ROCm installation path${ColourReset}")
endif()
# Set AMD Clang as default compiler
if (NOT DEFINED CMAKE_CXX_COMPILER)
  set(CMAKE_C_COMPILER ${ROCM_PATH}/bin/amdclang)
  set(CMAKE_CXX_COMPILER ${ROCM_PATH}/bin/amdclang++)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED On)

project(ROCJPEG-test)

# make test with CTest
enable_testing()
include(CTest)

# add find modules
list(APPEND CMAKE_MODULE_PATH ${ROCM_PATH}/share/rocjpeg/cmake)

# find rocDecode
find_package(rocJPEG QUIET)

if(NOT ROCJPEG_FOUND)
    message("-- ${Yellow}${PROJECT_NAME} requires rocJPEG. Install rocJPEG before running CTests")
endif()

add_test(
  NAME
    jpeg-decode-fmt-native
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${ROCM_PATH}/share/rocjpeg/samples/jpegDecode"
                              "${CMAKE_CURRENT_BINARY_DIR}/jpegDecode"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "jpegdecode"
            -i ${ROCM_PATH}/share/rocjpeg/images/
)

add_test(
  NAME
    jpeg-decode-fmt-yuv-planar
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${ROCM_PATH}/share/rocjpeg/samples/jpegDecode"
                              "${CMAKE_CURRENT_BINARY_DIR}/jpegDecode"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "jpegdecode"
            -i ${ROCM_PATH}/share/rocjpeg/images/ -fmt yuv_planar
)

add_test(
  NAME
    jpeg-decode-fmt-y
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${ROCM_PATH}/share/rocjpeg/samples/jpegDecode"
                              "${CMAKE_CURRENT_BINARY_DIR}/jpegDecode"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "jpegdecode"
            -i ${ROCM_PATH}/share/rocjpeg/images/ -fmt y
)

add_test(
  NAME
    jpeg-decode-fmt-rgb
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${ROCM_PATH}/share/rocjpeg/samples/jpegDecode"
                              "${CMAKE_CURRENT_BINARY_DIR}/jpegDecode"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "jpegdecode"
            -i ${ROCM_PATH}/share/rocjpeg/images/ -fmt rgb
)

add_test(
  NAME
    jpeg-decode-fmt-rgb-planar
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${ROCM_PATH}/share/rocjpeg/samples/jpegDecode"
                              "${CMAKE_CURRENT_BINARY_DIR}/jpegDecode"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "jpegdecode"
            -i ${ROCM_PATH}/share/rocjpeg/images/ -fmt rgb_planar
)

add_test(
  NAME
    jpeg-decode-threads-fmt-native
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${ROCM_PATH}/share/rocjpeg/samples/jpegDecodePerf"
                              "${CMAKE_CURRENT_BINARY_DIR}/jpegDecodePerf"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "jpegdecodeperf"
            -i ${ROCM_PATH}/share/rocjpeg/images/
)

add_test(
  NAME
    jpeg-decode-batch-fmt-native
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${ROCM_PATH}/share/rocjpeg/samples/jpegDecodeBatched"
                              "${CMAKE_CURRENT_BINARY_DIR}/jpegDecodeBatched"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "jpegdecodebatched"
            -i ${ROCM_PATH}/share/rocjpeg/images/
)

add_test(
  NAME
    jpeg-decode-crop-fmt-native
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${ROCM_PATH}/share/rocjpeg/samples/jpegDecode"
                              "${CMAKE_CURRENT_BINARY_DIR}/jpegDecode"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "jpegdecode"
            -i ${ROCM_PATH}/share/rocjpeg/images/ -crop 960,540,2880,1620
)

add_test(
  NAME
    jpeg-decode-crop-fmt-yuv-planar
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${ROCM_PATH}/share/rocjpeg/samples/jpegDecode"
                              "${CMAKE_CURRENT_BINARY_DIR}/jpegDecode"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "jpegdecode"
            -i ${ROCM_PATH}/share/rocjpeg/images/ -fmt yuv_planar -crop 960,540,2880,1620
)

add_test(
  NAME
    jpeg-decode-crop-fmt-y
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${ROCM_PATH}/share/rocjpeg/samples/jpegDecode"
                              "${CMAKE_CURRENT_BINARY_DIR}/jpegDecode"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "jpegdecode"
            -i ${ROCM_PATH}/share/rocjpeg/images/ -fmt y -crop 960,540,2880,1620
)

add_test(
  NAME
    jpeg-decode-crop-fmt-rgb
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${ROCM_PATH}/share/rocjpeg/samples/jpegDecode"
                              "${CMAKE_CURRENT_BINARY_DIR}/jpegDecode"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "jpegdecode"
            -i ${ROCM_PATH}/share/rocjpeg/images/ -fmt rgb -crop 960,540,2880,1620
)

add_test(
  NAME
    jpeg-decode-crop-fmt-rgb-planar
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${ROCM_PATH}/share/rocjpeg/samples/jpegDecode"
                              "${CMAKE_CURRENT_BINARY_DIR}/jpegDecode"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "jpegdecode"
            -i ${ROCM_PATH}/share/rocjpeg/images/ -fmt rgb_planar -crop 960,540,2880,1620
)

add_test(
  NAME
    jpeg-decode-crop-batch-fmt-native
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${ROCM_PATH}/share/rocjpeg/samples/jpegDecodeBatched"
                              "${CMAKE_CURRENT_BINARY_DIR}/jpegDecodeBatched"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "jpegdecodebatched"
            -i ${ROCM_PATH}/share/rocjpeg/images/ -crop 960,540,2880,1620
)