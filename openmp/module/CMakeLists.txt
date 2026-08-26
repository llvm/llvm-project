#//===----------------------------------------------------------------------===//
#//
#// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#// See https://llvm.org/LICENSE.txt for license information.
#// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#//
#//===----------------------------------------------------------------------===//

# Build the module files if a Fortran compiler is available.

configure_file(omp_lib.F90.var "${CMAKE_CURRENT_BINARY_DIR}/omp_lib.F90" @ONLY)
configure_file(omp_lib_impl.F90.var "${CMAKE_CURRENT_BINARY_DIR}/omp_lib_impl.F90" @ONLY)
configure_file(omp_lib.h.var "${RUNTIMES_OUTPUT_RESOURCE_MOD_DIR}/omp_lib.h" @ONLY)

# omp_lib.F90 declares the interfaces and creates both omp_lib.mod and
# omp_lib_kinds.mod; omp_lib_impl.F90 holds the bodies of the separate module
# procedures and creates omp_lib-omp_lib_impl.mod.
# TODO: Adding it to libomp ($<TARGET_OBJECTS:libomp-mod>) would allow implementing Fortran API in Fortran
add_library(libomp-mod OBJECT
  "${CMAKE_CURRENT_BINARY_DIR}/omp_lib.F90"
  "${CMAKE_CURRENT_BINARY_DIR}/omp_lib_impl.F90"
)
set_target_properties(libomp-mod PROPERTIES FOLDER "OpenMP/Fortran Modules")

# The following requests explicit building of the Fortran module files
# Workaround for gfortran to build modules with the
# omp_sched_monotonic integer parameter
if (CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
  target_compile_options(libomp-mod PRIVATE -fno-range-check)
endif ()

if (LIBOMP_TARGET_IS_GPU)
  target_compile_options(libomp-mod PRIVATE
    $<$<COMPILE_LANGUAGE:Fortran>:-nogpulib -flto>
  )
endif ()

flang_module_target(libomp-mod PUBLIC)

# flang_module_target(PUBLIC) emits the .mod files directly into the toolchain
# resource directory, but that directory is also Flang's intrinsic module
# directory, and when resolving the parent of a SUBMODULE Flang searches only
# the non-intrinsic module path. omp_lib.mod would therefore be invisible to
# omp_lib_impl.F90. Emit the module files into a private directory instead,
# where the submodule can find its parent, and publish the importable ones
# afterwards. omp_lib-omp_lib_impl.mod is only needed to compile further
# submodules of omp_lib, so it stays private.
set(libomp_mod_dir "${CMAKE_CURRENT_BINARY_DIR}/mod/${CMAKE_CFG_INTDIR}")
set_target_properties(libomp-mod PROPERTIES
  Fortran_MODULE_DIRECTORY "${libomp_mod_dir}"
)

set(libomp_public_mods
  "${RUNTIMES_OUTPUT_RESOURCE_MOD_DIR}/omp_lib.mod"
  "${RUNTIMES_OUTPUT_RESOURCE_MOD_DIR}/omp_lib_kinds.mod"
)
add_custom_command(OUTPUT ${libomp_public_mods}
  COMMAND "${CMAKE_COMMAND}" -E copy_if_different
          "${libomp_mod_dir}/omp_lib.mod"
          "${libomp_mod_dir}/omp_lib_kinds.mod"
          "${RUNTIMES_OUTPUT_RESOURCE_MOD_DIR}"
  DEPENDS libomp-mod
  COMMENT "Publishing omp_lib module files"
)

# libomp-mod is an OBJECT library, so the module files it writes are opaque to
# CMake. Dependees on omp_lib.mod must add a target-level dependency on this
# barrier to ensure the module files exist before they are used.
add_custom_target(libomp-mod-barrier ALL DEPENDS ${libomp_public_mods})
set_target_properties(libomp-mod-barrier PROPERTIES FOLDER "OpenMP/Fortran Modules")

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
