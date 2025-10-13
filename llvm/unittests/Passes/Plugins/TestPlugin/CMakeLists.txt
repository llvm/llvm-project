add_llvm_library(TestPlugin MODULE BUILDTREE_ONLY
  TestPlugin.cpp
)
set_target_properties(TestPlugin PROPERTIES FOLDER "Tests")

add_dependencies(PluginsTests TestPlugin)
