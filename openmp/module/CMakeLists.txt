#//===----------------------------------------------------------------------===//
#//
#// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#// See https://llvm.org/LICENSE.txt for license information.
#// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#//
#//===----------------------------------------------------------------------===//

# Build the module files if a Fortran compiler is available.
# Only LLVM_ENABLE_RUNTIMES=openmp is supported, LLVM_ENABLE_PROJECTS=openmp
# has been deprecated.

configure_file(omp_lib.F90.var "{CMAKE_CURRENT_BINARY_DIR}/omp_lib.F90" @ONLY)
configure_file(omp_lib.h.var "${CMAKE_CURRENT_BINARY_DIR}/../runtime/src/omp_lib.h" @ONLY)

# One compilation step creates both omp_lib.mod and omp_lib_kinds.mod. Only
# these files are used, the object file itself can be discarded.
# FIXME: Adding it to libomp.so would allow implementing Fortran API in Fortran
add_library(libomp-mod OBJECT
  "{CMAKE_CURRENT_BINARY_DIR}/omp_lib.F90"
)
set_target_properties(libomp-mod PROPERTIES FOLDER "OpenMP/Fortran Modules")

# The following requests explicit building of the Fortran module files
# Workaround for gfortran to build modules with the
# omp_sched_monotonic integer parameter
if (CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
  target_compile_options(libomp-mod PRIVATE -fno-range-check)
endif ()

flang_module_target(libomp-mod PUBLIC)
if (FORTRAN_MODULE_DEPS)
  add_dependencies(libomp-mod ${FORTRAN_MODULE_DEPS})
endif ()

set(destination "${LIBOMP_HEADERS_INSTALL_PATH}")
if (LIBOMP_MODULES_INSTALL_PATH)
  set(destination "${LIBOMP_MODULES_INSTALL_PATH}")
endif ()
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/../runtime/src/omp_lib.h"
  DESTINATION ${destination}
)
