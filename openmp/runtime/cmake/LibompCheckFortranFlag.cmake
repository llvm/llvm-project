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
    include(CheckFortranSourceCompiles)
    check_fortran_source_compiles("${fortran_source}" ${boolean} FAIL_REGEX "${failed_regexes}")
    set(${boolean} ${${boolean}} PARENT_SCOPE)
  endif()
endfunction()
