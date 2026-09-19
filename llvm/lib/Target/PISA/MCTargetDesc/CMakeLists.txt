add_llvm_component_library(LLVMPISADesc
  PISAInstPrinter.cpp
  PISAMCAsmInfo.cpp
  PISAMCTargetDesc.cpp

  LINK_COMPONENTS
  CodeGenTypes
  Core
  MC
  PISAInfo
  Support
  TargetParser

  ADD_TO_COMPONENT
  PISA
  )
