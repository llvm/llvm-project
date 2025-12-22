add_llvm_component_library(LLVMExtensions
  Extensions.cpp
  PassPlugin.cpp

  LINK_COMPONENTS
  Support
)
