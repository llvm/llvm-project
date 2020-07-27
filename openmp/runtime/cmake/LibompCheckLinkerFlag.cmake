#
#//===----------------------------------------------------------------------===//
#//
#// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#// See https://llvm.org/LICENSE.txt for license information.
#// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#//
#//===----------------------------------------------------------------------===//
#

# Checking a linker flag to build a shared library
# There is no real trivial way to do this in CMake, so we implement it here
# this will have ${boolean} = TRUE if the flag succeeds, otherwise FALSE.
function(libomp_check_linker_flag flag boolean)
  if(NOT DEFINED "${boolean}")
  set(retval TRUE)
  set(library_source
    "int foo(int a) { return a*a; }")
  set(cmake_source
    "cmake_minimum_required(VERSION 3.13.4)
     project(foo C)
     set(CMAKE_SHARED_LINKER_FLAGS \"${flag}\")
     add_library(foo SHARED src_to_link.c)")
  set(failed_regexes "[Ee]rror;[Uu]nknown;[Ss]kipping;LINK : warning")
  set(base_dir ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/link_flag_check_${boolean})
  file(MAKE_DIRECTORY ${base_dir})
  file(MAKE_DIRECTORY ${base_dir}/build)
  file(WRITE ${base_dir}/src_to_link.c "${library_source}")
  file(WRITE ${base_dir}/CMakeLists.txt "${cmake_source}")

  message(STATUS "Performing Test ${boolean}")
  try_compile(
    try_compile_result
    ${base_dir}/build
    ${base_dir}
    foo
    OUTPUT_VARIABLE OUTPUT)

  if(try_compile_result)
    foreach(regex IN LISTS failed_regexes)
      # Ignore the warning about the newer or unknown CUDA version.
      if(("${OUTPUT}" MATCHES ${regex}) AND NOT ("${OUTPUT}" MATCHES "Unknown CUDA version"))
        set(retval FALSE)
      endif()
    endforeach()
  else()
    set(retval FALSE)
  endif()

  if(${retval})
    set(${boolean} 1 CACHE INTERNAL "Test ${boolean}")
    message(STATUS "Performing Test ${boolean} - Success")
    file(APPEND ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeOutput.log
      "Performing C Linker Flag test ${boolean} succeeded with the following output:\n"
      "${OUTPUT}\n"
      "Source file was:\n${library_source}\n")
  else()
    set(${boolean} "" CACHE INTERNAL "Test ${boolean}")
    message(STATUS "Performing Test ${boolean} - Failed")
    file(APPEND ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeError.log
      "Performing C Linker Flag test ${boolean} failed with the following output:\n"
      "${OUTPUT}\n"
      "Source file was:\n${library_source}\n")
  endif()

  set(${boolean} ${retval} PARENT_SCOPE)
  endif()
endfunction()
