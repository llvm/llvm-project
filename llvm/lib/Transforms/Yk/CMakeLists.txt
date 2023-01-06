add_llvm_component_library(LLVMYkPasses
  BlockDisambiguate.cpp
  ControlPoint.cpp
  LivenessAnalysis.cpp
  StackMaps.cpp
  ShadowStack.cpp

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  Core
  Support
)
