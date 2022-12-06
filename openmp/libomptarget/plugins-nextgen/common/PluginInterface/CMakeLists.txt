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

  LINK_LIBS
    elf_common
)

# NOTE: Please don't move `MemoryManager` to `LINK_LIBS` above, even though it
# is an interface "library". Don't expect it to be treated as a "library" such
# that we just need to link it via CMake and its include path will be updated
# automatically. It is because `PluginInterface` is a static library here, and
# per CMake design (which makes sense), privately linked libraries have to be
# exported. However, there is no a typical way in CMake to export an interface
# library without install rules. We apparently don't need `MemoryManager`'s
# header out of the build tree.
target_include_directories(PluginInterface PRIVATE
  $<TARGET_PROPERTY:MemoryManager,INTERFACE_INCLUDE_DIRECTORIES>
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
