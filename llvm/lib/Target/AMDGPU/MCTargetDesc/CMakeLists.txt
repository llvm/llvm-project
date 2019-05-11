add_llvm_library(LLVMAMDGPUDesc
  AMDGPUAsmBackend.cpp
  AMDGPUELFObjectWriter.cpp
  AMDGPUELFStreamer.cpp
  AMDGPUInstPrinter.cpp
  AMDGPUMCAsmInfo.cpp
  AMDGPUMCCodeEmitter.cpp
  AMDGPUMCTargetDesc.cpp
  AMDGPUTargetStreamer.cpp
  R600MCCodeEmitter.cpp
  R600MCTargetDesc.cpp
  SIMCCodeEmitter.cpp
)

add_dependencies(LLVMAMDGPUDesc LLVMAMDGPUUtils)
