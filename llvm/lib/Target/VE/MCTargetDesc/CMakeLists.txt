add_llvm_component_library(LLVMVEDesc
  VEAsmBackend.cpp
  VEELFObjectWriter.cpp
  VEInstPrinter.cpp
  VEMCAsmInfo.cpp
  VEMCCodeEmitter.cpp
  VEMCExpr.cpp
  VEMCTargetDesc.cpp
  VETargetStreamer.cpp

  LINK_COMPONENTS
  MC
  VEInfo
  Support
  TargetParser

  ADD_TO_COMPONENT
  VE
  )
