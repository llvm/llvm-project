add_llvm_component_library(LLVMSPIRVDesc
  SPIRVBaseInfo.cpp
  SPIRVMCAsmInfo.cpp
  SPIRVMCTargetDesc.cpp
  SPIRVTargetStreamer.cpp
  SPIRVAsmBackend.cpp
  SPIRVMCCodeEmitter.cpp
  SPIRVInstPrinter.cpp

  LINK_COMPONENTS
  CodeGenTypes
  MC
  SPIRVInfo
  Support
  TargetParser

  ADD_TO_COMPONENT
  SPIRV
  )

# Static encoding tables and helper function redefinitions.
set_source_files_properties(
  SPIRVBaseInfo.cpp
  SPIRVMCCodeEmitter.cpp
  PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
