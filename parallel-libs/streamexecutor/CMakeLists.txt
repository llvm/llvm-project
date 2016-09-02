cmake_minimum_required(VERSION 3.1)

option(STREAM_EXECUTOR_UNIT_TESTS "enable unit tests" ON)
option(STREAM_EXECUTOR_ENABLE_DOXYGEN "enable StreamExecutor doxygen" ON)

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

    # Get the LLVM cxxflags by using llvm-config.
    #
    # This is necessary to get -fno-rtti if LLVM is compiled that way.
    execute_process(
        COMMAND
        "${LLVM_BINARY_DIR}/bin/llvm-config"
        --cxxflags
        OUTPUT_VARIABLE
        LLVM_CXXFLAGS
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${LLVM_CXXFLAGS}")

    # Find the libraries that correspond to the LLVM components
    # that we wish to use
    llvm_map_components_to_libnames(llvm_libs support symbolize)

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

# Insist on C++ 11 features.
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add warning flags.
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-unused-parameter")

add_subdirectory(lib)
add_subdirectory(examples)

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
