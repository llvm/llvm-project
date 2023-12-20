add_llvm_component_library(LLVMYkPasses
  BlockDisambiguate.cpp
  ControlPoint.cpp
  Linkage.cpp
  LivenessAnalysis.cpp
  StackMaps.cpp
  ShadowStack.cpp
  NoCallsInEntryBlocks.cpp
  SplitBlocksAfterCalls.cpp
  BasicBlockTracer.cpp

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  Analysis
  Core
  Support
)
