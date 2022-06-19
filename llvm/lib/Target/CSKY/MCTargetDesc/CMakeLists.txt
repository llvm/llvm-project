add_llvm_component_library(LLVMCSKYDesc
  CSKYAsmBackend.cpp
  CSKYELFObjectWriter.cpp
  CSKYELFStreamer.cpp
  CSKYInstPrinter.cpp
  CSKYMCAsmInfo.cpp
  CSKYMCExpr.cpp
  CSKYMCTargetDesc.cpp
  CSKYMCCodeEmitter.cpp
  CSKYTargetStreamer.cpp

  LINK_COMPONENTS
  CSKYInfo
  MC
  Support

  ADD_TO_COMPONENT
  CSKY
)
