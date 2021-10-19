add_llvm_component_library(LLVMMipsDisassembler
  MipsDisassembler.cpp

  LINK_COMPONENTS
  MCDisassembler
  MipsInfo
  Support

  ADD_TO_COMPONENT
  Mips
  )
