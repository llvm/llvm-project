add_llvm_library(DoublerPlugin MODULE BUILDTREE_ONLY
  DoublerPlugin.cpp
)
set_target_properties(DoublerPlugin PROPERTIES FOLDER "Tests")
add_dependencies(PluginsTests DoublerPlugin)
