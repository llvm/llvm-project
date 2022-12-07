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
add_llvm_library(PluginInterface PluginInterface.cpp GlobalHandler.cpp
  BUILDTREE_ONLY

  LINK_COMPONENTS
    Support

  LINK_LIBS PRIVATE
    elf_common
    MemoryManager
)

# Define the TARGET_NAME and DEBUG_PREFIX.
target_compile_definitions(PluginInterface PRIVATE
  TARGET_NAME="PluginInterface"
  DEBUG_PREFIX="PluginInterface"
)

target_include_directories(PluginInterface
  INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${LIBOMPTARGET_INCLUDE_DIR}
)

set_target_properties(PluginInterface PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CXX_VISIBILITY_PRESET protected)
