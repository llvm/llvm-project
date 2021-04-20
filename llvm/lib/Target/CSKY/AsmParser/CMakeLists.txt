add_llvm_component_library(LLVMCSKYAsmParser
  CSKYAsmParser.cpp

  LINK_COMPONENTS
  CSKYDesc
  CSKYInfo
  MC
  MCParser
  Support

  ADD_TO_COMPONENT
  CSKY
  )
