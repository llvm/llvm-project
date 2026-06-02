add_llvm_component_library(LLVMEZHDisassembler
  EZHDisassembler.cpp

  LINK_COMPONENTS
  CodeGenTypes
  EZHDesc
  EZHInfo
  MC
  MCDisassembler
  Support

  ADD_TO_COMPONENT
  EZH
  )
