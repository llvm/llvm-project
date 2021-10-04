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

add_library(elf_common OBJECT elf_common.cpp)

# Build elf_common with PIC to be able to link it with plugin shared libraries.
set_property(TARGET elf_common PROPERTY POSITION_INDEPENDENT_CODE ON)
llvm_update_compile_flags(elf_common)

set(LINK_LLVM_LIBS LLVMBinaryFormat LLVMObject LLVMSupport)
if (LLVM_LINK_LLVM_DYLIB)
  set(LINK_LLVM_LIBS LLVM)
endif()
target_link_libraries(elf_common INTERFACE ${LINK_LLVM_LIBS})
include_directories(${LIBOMPTARGET_LLVM_INCLUDE_DIRS})
add_dependencies(elf_common ${LINK_LLVM_LIBS})

# The code uses Debug.h, which requires threads support.
target_link_libraries(elf_common INTERFACE ${OPENMP_PTHREAD_LIB})

# Expose elf_common.h directory to the users of this library.
target_include_directories(elf_common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${LIBOMPTARGET_LLVM_INCLUDE_DIRS})

