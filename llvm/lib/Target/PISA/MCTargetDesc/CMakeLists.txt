add_llvm_component_library(LLVMPISADesc
  PISAMCAsmInfo.cpp
  PISAMCTargetDesc.cpp
  PISAInstPrinter.cpp

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
