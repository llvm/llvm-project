add_llvm_component_library(LLVMWebAssemblyDesc
  WebAssemblyAsmBackend.cpp
  WebAssemblyInstPrinter.cpp
  WebAssemblyMCAsmInfo.cpp
  WebAssemblyMCCodeEmitter.cpp
  WebAssemblyMCTargetDesc.cpp
  WebAssemblyTargetStreamer.cpp
  WebAssemblyWasmObjectWriter.cpp

  LINK_COMPONENTS
  MC
  Support
  WebAssemblyInfo
  WebAssemblyUtils

  ADD_TO_COMPONENT
  WebAssembly
)
