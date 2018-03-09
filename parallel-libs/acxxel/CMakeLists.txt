cmake_minimum_required(VERSION 3.1)

option(ACXXEL_ENABLE_UNIT_TESTS "enable acxxel unit tests" ON)
option(ACXXEL_ENABLE_MULTI_DEVICE_UNIT_TESTS "enable acxxel multi-device unit tests" OFF)
option(ACXXEL_ENABLE_EXAMPLES "enable acxxel examples" OFF)
option(ACXXEL_ENABLE_DOXYGEN "enable Doxygen for acxxel" OFF)
option(ACXXEL_ENABLE_CUDA "enable CUDA for acxxel" ON)
option(ACXXEL_ENABLE_OPENCL "enable OpenCL for acxxel" ON)

project(acxxel)

if(ACXXEL_ENABLE_CUDA)
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
find_library(CUDA_DRIVER_LIBRARY cuda)
if(NOT CUDA_DRIVER_LIBRARY)
    message(FATAL_ERROR "could not find libcuda, is the CUDA driver installed on your system?")
endif(NOT CUDA_DRIVER_LIBRARY)
set(ACXXEL_CUDA_SOURCES cuda_acxxel.cpp)
set(ACXXEL_CUDA_LIBRARIES ${CUDA_DRIVER_LIBRARY} ${CUDA_LIBRARIES})
endif(ACXXEL_ENABLE_CUDA)

if(ACXXEL_ENABLE_OPENCL)
find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})
set(ACXXEL_OPENCL_SOURCES opencl_acxxel.cpp)
set(ACXXEL_OPENCL_LIBRARIES ${OpenCL_LIBRARIES})
endif()

configure_file(config.h.in config.h)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

# Insist on C++ 11 features.
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add warning flags.
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall -Wextra")

add_library(
    acxxel
    acxxel.cpp
    ${ACXXEL_CUDA_SOURCES}
    ${ACXXEL_OPENCL_SOURCES})
target_link_libraries(
    acxxel
    ${ACXXEL_CUDA_LIBRARIES}
    ${ACXXEL_OPENCL_LIBRARIES})

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

if(ACXXEL_ENABLE_EXAMPLES)
    add_subdirectory(examples)
endif()

if(ACXXEL_ENABLE_UNIT_TESTS)
    enable_testing()
    find_package(GTest REQUIRED)
    include_directories(${GTEST_INCLUDE_DIRS})
    find_package(Threads)
    add_subdirectory(tests)
endif()

if(ACXXEL_ENABLE_DOXYGEN)
    find_package(Doxygen REQUIRED)
    configure_file(Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
    add_custom_target(
        acxxel-doc
        ${DOXYGEN_EXECUTABLE}
        ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
        WORKING_DIRECTORY
        ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT
        "Generating acxxel API documentation with Doxygen"
        VERBATIM)
endif()
