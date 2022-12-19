add_llvm_component_library(LLVMAVRDisassembler
  AVRDisassembler.cpp

  LINK_COMPONENTS
  MCDisassembler
  MC
  Support
  AVRInfo

  ADD_TO_COMPONENT
  AVR
)

