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

  LINK_COMPONENTS
  AMDGPUInfo
  AMDGPUUtils
  BinaryFormat
  CodeGenTypes
  Core
  MC
  Support
  TargetParser

  ADD_TO_COMPONENT
  AMDGPU
  )
