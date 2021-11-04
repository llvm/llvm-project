add_llvm_component_library(LLVMYkPasses
  BlockDisambiguate.cpp
  ControlPoint.cpp

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  Core
  Support
)
