add_llvm_component_library(LLVMRISCVDesc
  RISCVAsmBackend.cpp
  RISCVBaseInfo.cpp
  RISCVELFObjectWriter.cpp
  RISCVInstPrinter.cpp
  RISCVMCAsmInfo.cpp
  RISCVMCCodeEmitter.cpp
  RISCVMCExpr.cpp
  RISCVMCTargetDesc.cpp
  RISCVMatInt.cpp
  RISCVTargetStreamer.cpp
  RISCVELFStreamer.cpp

  LINK_COMPONENTS
  MC
  RISCVInfo
  Support

  ADD_TO_COMPONENT
  RISCV
)
