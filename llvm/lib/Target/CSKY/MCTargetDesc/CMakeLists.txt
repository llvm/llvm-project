add_llvm_component_library(LLVMCSKYDesc
  CSKYAsmBackend.cpp
  CSKYELFObjectWriter.cpp
  CSKYELFStreamer.cpp
  CSKYInstPrinter.cpp
  CSKYMCAsmInfo.cpp
  CSKYMCTargetDesc.cpp
  CSKYMCCodeEmitter.cpp
  CSKYTargetStreamer.cpp

  LINK_COMPONENTS
  CSKYInfo
  CodeGenTypes
  MC
  Option
  Support
  TargetParser

  ADD_TO_COMPONENT
  CSKY
  )
