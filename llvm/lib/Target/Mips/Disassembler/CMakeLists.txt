add_llvm_component_library(LLVMMipsDisassembler
  MipsDisassembler.cpp

  LINK_COMPONENTS
  MCDisassembler
  MC
  MipsDesc
  MipsInfo
  Support

  ADD_TO_COMPONENT
  Mips
  )
