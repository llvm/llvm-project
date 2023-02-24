add_llvm_component_library(LLVMWebAssemblyAsmParser
  WebAssemblyAsmParser.cpp
  WebAssemblyAsmTypeCheck.cpp

  LINK_COMPONENTS
  MC
  MCParser
  WebAssemblyDesc
  WebAssemblyInfo
  Support
  TargetParser

  ADD_TO_COMPONENT
  WebAssembly
  )
