add_llvm_component_library(LLVMAIRDesc
  AIRContainerObjectWriter.cpp
  AIRMCTargetDesc.cpp

  LINK_COMPONENTS
  AIRInfo
  MC
  Support
  TargetParser

  ADD_TO_COMPONENT
  AIR
  )

add_dependencies(LLVMAIRDesc AIRCommonTableGen)
