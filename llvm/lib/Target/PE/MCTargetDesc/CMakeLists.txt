add_llvm_component_library(LLVMPEDesc
  PEMCTargetDesc.cpp
  PEMCAsmInfo.cpp
  PEInstPrinter.cpp
  
  LINK_COMPONENTS
  MC
  PEInfo
  Support
  TargetParser
  
  ADD_TO_COMPONENT
  PE
)