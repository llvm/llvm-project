add_llvm_component_library(LLVMSuperHDesc
  SuperHMCTargetDesc.cpp

  LINK_COMPONENTS
  MC
  MCDisassembler
  Support
  TargetParser

  ADD_TO_COMPONENT
  SuperH
)
