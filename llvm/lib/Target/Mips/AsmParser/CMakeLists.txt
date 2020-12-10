add_llvm_component_library(LLVMMipsAsmParser
  MipsAsmParser.cpp

  LINK_COMPONENTS
  MC
  MCParser
  MipsDesc
  MipsInfo
  Support

  ADD_TO_COMPONENT
  Mips
  )
