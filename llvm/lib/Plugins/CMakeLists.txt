add_llvm_component_library(LLVMPlugins
  PassPlugin.cpp

  LINK_COMPONENTS
  Support
)
