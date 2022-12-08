##===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##
#
# Common ELF functionality for target plugins
#
##===----------------------------------------------------------------------===##

# NOTE: Don't try to build `elf_common` using `add_llvm_library`.
# See openmp/libomptarget/plugins-nextgen/common/PluginInterface/CMakeLists.txt
# for more explanation.
add_library(elf_common OBJECT elf_common.cpp ELFSymbols.cpp)

# This is required when using LLVM libraries.
llvm_update_compile_flags(elf_common)

if (LLVM_LINK_LLVM_DYLIB)
  set(llvm_libs LLVM)
else()
  llvm_map_components_to_libnames(llvm_libs BinaryFormat Object Support)
endif()

target_link_libraries(elf_common PUBLIC ${llvm_libs} ${OPENMP_PTHREAD_LIB})

# Build elf_common with PIC to be able to link it with plugin shared libraries.
set_property(TARGET elf_common PROPERTY POSITION_INDEPENDENT_CODE ON)

# Expose elf_common.h directory to the users of this library.
target_include_directories(elf_common
  INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${LIBOMPTARGET_INCLUDE_DIR}
)
