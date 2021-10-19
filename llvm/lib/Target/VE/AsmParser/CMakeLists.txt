add_llvm_component_library(LLVMVEAsmParser
  VEAsmParser.cpp

  LINK_COMPONENTS
  MC
  MCParser
  VEDesc
  VEInfo
  Support

  ADD_TO_COMPONENT
  VE
  )
