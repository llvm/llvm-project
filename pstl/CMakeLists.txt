#===-- CMakeLists.txt ----------------------------------------------------===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is dual licensed under the MIT and the University of Illinois Open
# Source Licenses. See LICENSE.TXT for details.
#
#===----------------------------------------------------------------------===##
cmake_minimum_required(VERSION 3.4.3)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(PARALLELSTL_VERSION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/include/pstl/internal/pstl_config.h")
file(STRINGS "${PARALLELSTL_VERSION_FILE}" PARALLELSTL_VERSION_SOURCE REGEX "#define PSTL_VERSION .*$")
string(REGEX MATCH "#define PSTL_VERSION (.*)$" PARALLELSTL_VERSION_SOURCE "${PARALLELSTL_VERSION_SOURCE}")
math(EXPR VERSION_MAJOR "${PARALLELSTL_VERSION_SOURCE} / 100")
math(EXPR VERSION_MINOR "${PARALLELSTL_VERSION_SOURCE} % 100")

project(ParallelSTL VERSION ${VERSION_MAJOR}.${VERSION_MINOR} LANGUAGES CXX)

option(PARALLELSTL_USE_PARALLEL_POLICIES "Enable parallel policies" ON)
set(PARALLELSTL_BACKEND "tbb" CACHE STRING "Threading backend; defaults to TBB")

include(CMakePackageConfigHelpers)

if (NOT TBB_DIR)
    get_filename_component(PSTL_DIR_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
    string(REPLACE pstl tbb TBB_DIR_NAME ${PSTL_DIR_NAME})
    if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../${TBB_DIR_NAME}/cmake")
        get_filename_component(TBB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../${TBB_DIR_NAME}/cmake" ABSOLUTE)
    endif()
endif()

add_library(ParallelSTL INTERFACE)
add_library(pstl::ParallelSTL ALIAS ParallelSTL)

if (PARALLELSTL_USE_PARALLEL_POLICIES)
    if (PARALLELSTL_BACKEND STREQUAL "tbb")
        find_package(TBB 2018 REQUIRED tbb OPTIONAL_COMPONENTS tbbmalloc)
        message(STATUS "Parallel STL uses TBB ${TBB_VERSION} (interface version: ${TBB_INTERFACE_VERSION})")
        target_link_libraries(ParallelSTL INTERFACE TBB::tbb)
    else()
        if (TARGET ${PARALLELSTL_BACKEND})
            target_link_libraries(ParallelSTL INTERFACE ${PARALLELSTL_BACKEND})
        else()
            find_package(${PARALLELSTL_BACKEND} REQUIRED)
            target_link_libraries(ParallelSTL INTERFACE ${${PARALLELSTL_BACKEND}_IMPORTED_TARGETS})
        endif()
    endif()
else()
    target_compile_definitions(ParallelSTL INTERFACE PSTL_USE_PARALLEL_POLICIES=0)
endif()

target_include_directories(ParallelSTL
    INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/ParallelSTLConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion)

configure_file(
    ParallelSTLConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/ParallelSTLConfig.cmake
    @ONLY)

export(TARGETS ParallelSTL NAMESPACE pstl:: FILE ParallelSTLTargets.cmake)
export(PACKAGE ParallelSTL)

enable_testing()
add_subdirectory(test)
