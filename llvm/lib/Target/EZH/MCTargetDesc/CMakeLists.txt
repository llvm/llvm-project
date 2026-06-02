add_llvm_component_library(LLVMEZHDesc
  EZHAsmBackend.cpp
  EZHELFObjectWriter.cpp
  EZHInstPrinter.cpp
  EZHMCAsmInfo.cpp
  EZHMCCodeEmitter.cpp
  EZHMCTargetDesc.cpp

  LINK_COMPONENTS
  EZHInfo
  MC
  MCDisassembler
  Support
  TargetParser

  ADD_TO_COMPONENT
  EZH
)
