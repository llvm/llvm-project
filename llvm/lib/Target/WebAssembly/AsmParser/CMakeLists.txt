add_llvm_component_library(LLVMWebAssemblyAsmParser
  WebAssemblyAsmParser.cpp
  WebAssemblyAsmTypeCheck.cpp

  LINK_COMPONENTS
  CodeGen
  MC
  MCParser
  Support
  TargetParser
  WebAssemblyDesc
  WebAssemblyInfo

  ADD_TO_COMPONENT
  WebAssembly
  )
