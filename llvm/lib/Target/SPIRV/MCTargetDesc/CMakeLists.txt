add_llvm_component_library(LLVMSPIRVDesc
  SPIRVMCAsmInfo.cpp
  SPIRVMCTargetDesc.cpp
  SPIRVTargetStreamer.cpp
  SPIRVAsmBackend.cpp
  SPIRVMCCodeEmitter.cpp
  SPIRVObjectTargetWriter.cpp
  SPIRVInstPrinter.cpp

  LINK_COMPONENTS
  MC
  SPIRVInfo
  Support

  ADD_TO_COMPONENT
  SPIRV
  )
