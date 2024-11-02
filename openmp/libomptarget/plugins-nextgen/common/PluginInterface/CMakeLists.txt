##===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##
#
# Common parts which can be used by all plugins
#
##===----------------------------------------------------------------------===##

# Plugin Interface library.
add_library(PluginInterface OBJECT PluginInterface.cpp GlobalHandler.cpp)

# Define the TARGET_NAME.
add_definitions("-DTARGET_NAME=PluginInterface")

# Define the DEBUG_PREFIX.
add_definitions(-DDEBUG_PREFIX="PluginInterface")

set_target_properties(PluginInterface PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CXX_VISIBILITY_PRESET protected)
llvm_update_compile_flags(PluginInterface)
set(LINK_LLVM_LIBS LLVMSupport)
if (LLVM_LINK_LLVM_DYLIB)
  set(LINK_LLVM_LIBS LLVM)
endif()
target_link_libraries(PluginInterface INTERFACE ${LINK_LLVM_LIBS} PRIVATE elf_common MemoryManager)
add_dependencies(PluginInterface ${LINK_LLVM_LIBS})

target_include_directories(PluginInterface INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(PluginInterface PRIVATE ${LIBOMPTARGET_INCLUDE_DIR})
