add_llvm_component_library(LLVMLanaiDisassembler
  LanaiDisassembler.cpp

  LINK_COMPONENTS
  LanaiDesc
  LanaiInfo
  MC
  MCDisassembler
  Support

  ADD_TO_COMPONENT
  Lanai
  )
