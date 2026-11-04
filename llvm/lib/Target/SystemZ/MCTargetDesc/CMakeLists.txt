add_llvm_component_library(LLVMSystemZDesc
  SystemZELFObjectWriter.cpp
  SystemZGNUInstPrinter.cpp
  SystemZGOFFObjectWriter.cpp
  SystemZHLASMAsmStreamer.cpp
  SystemZHLASMInstPrinter.cpp
  SystemZInstPrinterCommon.cpp
  SystemZMCAsmBackend.cpp
  SystemZMCAsmInfo.cpp
  SystemZMCCodeEmitter.cpp
  SystemZMCTargetDesc.cpp
  SystemZTargetStreamer.cpp

  LINK_COMPONENTS
  CodeGenTypes
  MC
  Support
  SystemZInfo

  ADD_TO_COMPONENT
  SystemZ
  )
