#
#//===----------------------------------------------------------------------===//
#//
#// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#// See https://llvm.org/LICENSE.txt for license information.
#// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#//
#//===----------------------------------------------------------------------===//
#

# Checking a fortran compiler flag
# There is no real trivial way to do this in CMake, so we implement it here
# this will have ${boolean} = TRUE if the flag succeeds, otherwise false.
function(libomp_check_fortran_flag flag boolean)
  if(NOT DEFINED "${boolean}")
    set(retval TRUE)
    set(fortran_source
"      program hello
           print *, \"Hello World!\"
      end program hello")

  set(failed_regexes "[Ee]rror;[Uu]nknown;[Ss]kipping")
  if(CMAKE_VERSION VERSION_GREATER 3.1 OR CMAKE_VERSION VERSION_EQUAL 3.1)
    include(CheckFortranSourceCompiles)
    check_fortran_source_compiles("${fortran_source}" ${boolean} FAIL_REGEX "${failed_regexes}")
    set(${boolean} ${${boolean}} PARENT_SCOPE)
    return()
  else()
    # Our manual check for cmake versions that don't have CheckFortranSourceCompiles
    set(base_dir ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/fortran_flag_check)
    file(MAKE_DIRECTORY ${base_dir})
    file(WRITE ${base_dir}/fortran_source.f "${fortran_source}")

    message(STATUS "Performing Test ${boolean}")
    execute_process(
      COMMAND ${CMAKE_Fortran_COMPILER} "${flag}" ${base_dir}/fortran_source.f
      WORKING_DIRECTORY ${base_dir}
      RESULT_VARIABLE exit_code
      OUTPUT_VARIABLE OUTPUT
      ERROR_VARIABLE OUTPUT
    )

    if(${exit_code} EQUAL 0)
      foreach(regex IN LISTS failed_regexes)
        if("${OUTPUT}" MATCHES ${regex})
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
        "Performing Fortran Compiler Flag test ${boolean} succeeded with the following output:\n"
        "${OUTPUT}\n"
        "Source file was:\n${fortran_source}\n")
    else()
      set(${boolean} "" CACHE INTERNAL "Test ${boolean}")
      message(STATUS "Performing Test ${boolean} - Failed")
      file(APPEND ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeError.log
        "Performing Fortran Compiler Flag test ${boolean} failed with the following output:\n"
        "${OUTPUT}\n"
        "Source file was:\n${fortran_source}\n")
    endif()
  endif()

  set(${boolean} ${retval} PARENT_SCOPE)
  endif()
endfunction()
