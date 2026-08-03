add_llvm_component_library(LLVMPISADesc
  PISAEnum.cpp
  PISAInstPrinter.cpp
  PISAMCAsmInfo.cpp
  PISAMCExpr.cpp
  PISAMCTargetDesc.cpp
  PISARegEncoder.cpp
  PISATargetStreamer.cpp

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

