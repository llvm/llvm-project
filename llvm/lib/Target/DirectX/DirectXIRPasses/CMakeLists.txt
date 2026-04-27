add_llvm_component_library(LLVMDirectXIRPasses
  DXILDebugInfo.cpp
  PointerTypeAnalysis.cpp

  LINK_COMPONENTS
  Core
  Support

  ADD_TO_COMPONENT
  DirectX
  )
