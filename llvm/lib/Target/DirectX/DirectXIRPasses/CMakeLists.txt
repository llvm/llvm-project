add_llvm_component_library(LLVMDirectXIRPasses
  DXILAttributes.cpp
  DXILDebugInfo.cpp
  PointerTypeAnalysis.cpp

  LINK_COMPONENTS
  Core
  Support
  TransformUtils

  ADD_TO_COMPONENT
  DirectX
  )
