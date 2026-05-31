add_llvm_component_library(LLVMSuperHDesc
  SuperHMCTargetDesc.cpp
  SuperHMCAsmInfo.cpp
  SuperHInstPrinter.cpp

  LINK_COMPONENTS
  MC
  SuperHInfo
  Support
  TargetParser

  ADD_TO_COMPONENT
  SuperH
)
