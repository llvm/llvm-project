add_llvm_component_library(LLVMAVRDisassembler
  AVRDisassembler.cpp

  LINK_COMPONENTS
  AVRInfo
  CodeGen
  MC
  MCDisassembler
  Support

  ADD_TO_COMPONENT
  AVR
  )

