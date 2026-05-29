add_llvm_component_library(LLVMMetalDesc
  MetalContainerObjectWriter.cpp
  MetalMCTargetDesc.cpp

  LINK_COMPONENTS
  MetalInfo
  MC
  Support
  TargetParser

  ADD_TO_COMPONENT
  Metal
  )

add_dependencies(LLVMMetalDesc MetalCommonTableGen)
