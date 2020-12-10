add_llvm_component_library(LLVMWebAssemblyDisassembler
  WebAssemblyDisassembler.cpp

  LINK_COMPONENTS
  WebAssemblyDesc
  MCDisassembler
  WebAssemblyInfo
  Support
  MC

  ADD_TO_COMPONENT
  WebAssembly
  )
