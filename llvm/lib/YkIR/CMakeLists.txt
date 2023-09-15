add_llvm_component_library(LLVMYkIR
  YkIRWriter.cpp

  LINK_COMPONENTS
  Core
  MC
  Support
  )
