add_llvm_component_library(LLVMWebAssemblyAsmParser
  WebAssemblyAsmParser.cpp
  WebAssemblyAsmTypeCheck.cpp

  LINK_COMPONENTS
  MC
  MCParser
  WebAssemblyInfo
  WebAssemblyUtils
  Support
  TargetParser

  ADD_TO_COMPONENT
  WebAssembly
  )
