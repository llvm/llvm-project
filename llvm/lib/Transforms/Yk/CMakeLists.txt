add_llvm_component_library(LLVMYkPasses
  BlockDisambiguate.cpp
  ControlPoint.cpp
  LivenessAnalysis.cpp
  StackMaps.cpp

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  Core
  Support
)
