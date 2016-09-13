cmake_minimum_required(VERSION 3.1)

option(STREAM_EXECUTOR_UNIT_TESTS "enable unit tests" ON)
option(STREAM_EXECUTOR_ENABLE_DOXYGEN "enable StreamExecutor doxygen" ON)
option(STREAM_EXECUTOR_ENABLE_CONFIG_TOOL "enable building streamexecutor-config tool" ON)

# First find includes relative to the streamexecutor top-level source path.
include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# If we are not building as part of LLVM, build StreamExecutor as a standalone
# project using LLVM as an external library:
string(
    COMPARE
    EQUAL
    "${CMAKE_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}"
    STREAM_EXECUTOR_STANDALONE)

if(STREAM_EXECUTOR_STANDALONE)
    project(StreamExecutor)

    find_package(LLVM REQUIRED CONFIG)
    message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
    message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

    include_directories(${LLVM_INCLUDE_DIRS})
    add_definitions(${LLVM_DEFINITIONS})

    # If LLVM does not have RTTI, don't use it here either.
    if (NOT LLVM_ENABLE_RTTI)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
    endif()

    set(LLVM_CMAKE_PATH "${LLVM_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX}/cmake/llvm")
    list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_PATH}")
    include(AddLLVM)

    if(STREAM_EXECUTOR_UNIT_TESTS)
        enable_testing()
        find_package(GTest REQUIRED)
        include_directories(${GTEST_INCLUDE_DIRS})
        find_package(Threads REQUIRED)
    endif()
else(NOT STREAM_EXECUTOR_STANDALONE)
    if(STREAM_EXECUTOR_UNIT_TESTS)
        include_directories(
            "${LLVM_MAIN_SRC_DIR}/utils/unittest/googletest/include")
    endif()
endif(STREAM_EXECUTOR_STANDALONE)

# Find the libraries that correspond to the LLVM components
# that we wish to use
llvm_map_components_to_libnames(llvm_libs support symbolize)

# Insist on C++ 11 features.
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add warning flags.
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-unused-parameter")

add_subdirectory(lib)
add_subdirectory(examples)

if(STREAM_EXECUTOR_UNIT_TESTS)
    add_subdirectory(unittests)
endif()

if(STREAM_EXECUTOR_ENABLE_CONFIG_TOOL )
    add_subdirectory(tools/streamexecutor-config)
endif(STREAM_EXECUTOR_ENABLE_CONFIG_TOOL )

install(DIRECTORY include/ DESTINATION include)

if (STREAM_EXECUTOR_ENABLE_DOXYGEN)
    find_package(Doxygen REQUIRED)
    configure_file(Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
    add_custom_target(
        doc
        ${DOXYGEN_EXECUTABLE}
        ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
        WORKING_DIRECTORY
        ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT
        "Generating API documentation with Doxygen"
        VERBATIM)
endif(STREAM_EXECUTOR_ENABLE_DOXYGEN)
