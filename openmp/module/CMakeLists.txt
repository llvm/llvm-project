#//===----------------------------------------------------------------------===//
#//
#// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#// See https://llvm.org/LICENSE.txt for license information.
#// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#//
#//===----------------------------------------------------------------------===//

# Build the module files if a Fortran compiler is available.

configure_file(omp_lib.F90.var "${CMAKE_CURRENT_BINARY_DIR}/omp_lib.F90" @ONLY)
configure_file(omp_lib.h.var "${RUNTIMES_OUTPUT_RESOURCE_MOD_DIR}/omp_lib.h" @ONLY)

# One compilation step creates both, omp_lib.mod and omp_lib_kinds.mod. Only
# these files are used, the object file itself can be discarded.
# TODO: Adding it to libomp ($<TARGET_OBJECTS:libomp-mod>) would allow implementing Fortran API in Fortran
add_library(libomp-mod OBJECT
  "${CMAKE_CURRENT_BINARY_DIR}/omp_lib.F90"
)
set_target_properties(libomp-mod PROPERTIES FOLDER "OpenMP/Fortran Modules")

# The following requests explicit building of the Fortran module files
# Workaround for gfortran to build modules with the
# omp_sched_monotonic integer parameter
if (CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
  target_compile_options(libomp-mod PRIVATE -fno-range-check)
endif ()

if ("${LLVM_DEFAULT_TARGET_TRIPLE}" MATCHES "^amdgcn|^nvptx")
  target_compile_options(libomp-mod PRIVATE
    $<$<COMPILE_LANGUAGE:Fortran>:-nogpulib -flto>
  )
endif ()

flang_module_target(libomp-mod PUBLIC)
add_dependencies(libomp-mod ${RUNTIMES_FORTRAN_BUILD_DEPS})

install(FILES
  "${RUNTIMES_OUTPUT_RESOURCE_MOD_DIR}/omp_lib.h"
  DESTINATION "${RUNTIMES_INSTALL_RESOURCE_MOD_PATH}"
    COMPONENT openmp
)

openmp_register_meta_dep(libomp-mod)
