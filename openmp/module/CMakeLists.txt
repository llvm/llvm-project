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

if ("flang-rt" IN_LIST LLVM_ENABLE_RUNTIMES)
  set(flang_rt_module_stamp
    "${CMAKE_BINARY_DIR}/flang-rt/lib/runtime/flang_rt.mod.stamp"
  )
  set_property(SOURCE "${CMAKE_CURRENT_BINARY_DIR}/omp_lib.F90"
    APPEND PROPERTY OBJECT_DEPENDS "${flang_rt_module_stamp}"
  )
endif ()

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
  COMPONENT libomp-mod
  EXCLUDE_FROM_ALL
)

if (NOT CMAKE_CONFIGURATION_TYPES)
  add_custom_target(install-libomp-mod
    DEPENDS libomp-mod
    COMMAND "${CMAKE_COMMAND}" --install "${CMAKE_BINARY_DIR}"
            --component libomp-mod)
  add_custom_target(install-libomp-mod-stripped
    DEPENDS libomp-mod
    COMMAND "${CMAKE_COMMAND}" --install "${CMAKE_BINARY_DIR}"
            --component libomp-mod --strip)
endif ()
