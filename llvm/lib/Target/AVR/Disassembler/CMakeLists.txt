add_llvm_component_library(LLVMAVRDisassembler
  AVRDisassembler.cpp

  LINK_COMPONENTS
  MCDisassembler
  AVRInfo
  Support

  ADD_TO_COMPONENT
  AVR
)

