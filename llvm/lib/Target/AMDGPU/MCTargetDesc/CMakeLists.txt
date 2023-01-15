add_llvm_component_library(LLVMAMDGPUDesc
  AMDGPUAsmBackend.cpp
  AMDGPUELFObjectWriter.cpp
  AMDGPUELFStreamer.cpp
  AMDGPUInstPrinter.cpp
  AMDGPUMCAsmInfo.cpp
  AMDGPUMCCodeEmitter.cpp
  AMDGPUMCTargetDesc.cpp
  AMDGPUTargetStreamer.cpp
  R600InstPrinter.cpp
  R600MCCodeEmitter.cpp
  R600MCTargetDesc.cpp
  SIMCCodeEmitter.cpp

  LINK_COMPONENTS
  Core
  MC
  AMDGPUInfo
  AMDGPUUtils
  Support
  BinaryFormat
  TargetParser

  ADD_TO_COMPONENT
  AMDGPU
)

add_dependencies(LLVMAMDGPUDesc LLVMAMDGPUUtils)
