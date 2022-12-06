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

add_llvm_library(elf_common elf_common.cpp ELFSymbols.cpp
  BUILDTREE_ONLY

  LINK_COMPONENTS
    BinaryFormat
    Object
    Support

  LINK_LIBS
    ${OPENMP_PTHREAD_LIB}
)

# Build elf_common with PIC to be able to link it with plugin shared libraries.
set_property(TARGET elf_common PROPERTY POSITION_INDEPENDENT_CODE ON)

# Expose elf_common.h directory to the users of this library.
target_include_directories(elf_common
  INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${LIBOMPTARGET_INCLUDE_DIR}
)
