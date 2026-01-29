#//===----------------------------------------------------------------------===//
#//
#// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#// See https://llvm.org/LICENSE.txt for license information.
#// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#//
#//===----------------------------------------------------------------------===//

include(LibompCheckFortranFlag)

set(LIBOMP_FFLAGS "" CACHE STRING
  "Appended user specified Fortran compiler flags.  These are only used if LIBOMP_FORTRAN_MODULES==TRUE.")

# Enabling Fortran if it is needed
if (LIBOMP_FORTRAN_MODULES)
  enable_language(Fortran)

  libomp_check_fortran_flag(-m32 LIBOMP_HAVE_M32_FORTRAN_FLAG)
endif ()

# Building the Fortran module files
# One compilation step creates both omp_lib.mod and omp_lib_kinds.mod
configure_file(omp_lib.F90.var omp_lib.F90 @ONLY)
configure_file(omp_lib.h.var "${CMAKE_CURRENT_BINARY_DIR}/../runtime/src/omp_lib.h" @ONLY)

set(BUILD_FORTRAN_MODULES False)
if (LIBOMP_FORTRAN_MODULES_COMPILER)
  # If libomp is built as an LLVM runtime and the flang compiler is available,
  # compile the Fortran module files.
  message(STATUS "configuring openmp to build Fortran module files using '${LIBOMP_FORTRAN_MODULES_COMPILER}'")
  set(LIBOMP_FORTRAN_SOURCE_FILE omp_lib.F90)
  add_custom_target(libomp-mod ALL DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/../runtime/src/omp_lib.mod" "${CMAKE_CURRENT_BINARY_DIR}/../runtime/src/omp_lib_kinds.mod")
  add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/../runtime/src/omp_lib.mod" "${CMAKE_CURRENT_BINARY_DIR}/../runtime/src/omp_lib_kinds.mod"
    COMMAND ${LIBOMP_FORTRAN_MODULES_COMPILER} -cpp -fsyntax-only ${LIBOMP_FORTRAN_SOURCE_FILE} "-J${CMAKE_CURRENT_BINARY_DIR}/../runtime/src"
    DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/${LIBOMP_FORTRAN_SOURCE_FILE}"
  )
  set(BUILD_FORTRAN_MODULES True)
elseif (LIBOMP_FORTRAN_MODULES)
  # The following requests explicit building of the Fortran module files
  # Workaround for gfortran to build modules with the
  # omp_sched_monotonic integer parameter
  if (CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
    set(ADDITIONAL_Fortran_FLAGS "-fno-range-check")
  endif ()
  add_custom_target(libomp-mod ALL DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/../runtime/src/omp_lib.mod" "${CMAKE_CURRENT_BINARY_DIR}/../runtime/src/omp_lib_kinds.mod")
  set_target_properties(libomp-mod PROPERTIES FOLDER "OpenMP/Misc")
  libomp_get_fflags(LIBOMP_CONFIGURED_FFLAGS)
  if (CMAKE_Fortran_COMPILER_SUPPORTS_F90)
    set(LIBOMP_FORTRAN_SOURCE_FILE omp_lib.F90)
  else ()
    message(FATAL_ERROR "Fortran module build requires Fortran 90 compiler")
  endif ()
  add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/../runtime/src/omp_lib.mod" "${CMAKE_CURRENT_BINARY_DIR}/../runtime/src/omp_lib_kinds.mod"
    COMMAND ${CMAKE_Fortran_COMPILER} -c ${ADDITIONAL_Fortran_FLAGS}
            ${LIBOMP_CONFIGURED_FFLAGS} ${LIBOMP_FORTRAN_SOURCE_FILE} "-J${CMAKE_CURRENT_BINARY_DIR}/../runtime/src"
    DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/${LIBOMP_FORTRAN_SOURCE_FILE}"
  )
  set_property(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/../runtime/src" PROPERTY ADDITIONAL_MAKE_CLEAN_FILES "omp_lib${CMAKE_C_OUTPUT_EXTENSION}")
  set(BUILD_FORTRAN_MODULES True)
endif ()


if (BUILD_FORTRAN_MODULES)
  set(destination "${LIBOMP_HEADERS_INSTALL_PATH}")
  if (LIBOMP_MODULES_INSTALL_PATH)
    set(destination "${LIBOMP_MODULES_INSTALL_PATH}")
  endif ()
  install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/../runtime/src/omp_lib.mod"
    "${CMAKE_CURRENT_BINARY_DIR}/../runtime/src/omp_lib_kinds.mod"
    "${CMAKE_CURRENT_BINARY_DIR}/../runtime/src/omp_lib.h"
    DESTINATION ${destination}
  )
endif ()
