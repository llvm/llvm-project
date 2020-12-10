add_llvm_component_library(LLVMRISCVDesc
  RISCVAsmBackend.cpp
  RISCVELFObjectWriter.cpp
  RISCVInstPrinter.cpp
  RISCVMCAsmInfo.cpp
  RISCVMCCodeEmitter.cpp
  RISCVMCExpr.cpp
  RISCVMCTargetDesc.cpp
  RISCVTargetStreamer.cpp
  RISCVELFStreamer.cpp

  LINK_COMPONENTS
  MC
  RISCVInfo
  RISCVUtils
  Support

  ADD_TO_COMPONENT
  RISCV
)
