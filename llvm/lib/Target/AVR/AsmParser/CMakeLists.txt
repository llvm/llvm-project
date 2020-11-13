add_llvm_component_library(LLVMAVRAsmParser
  AVRAsmParser.cpp

  LINK_COMPONENTS
  MC
  MCParser
  AVRDesc
  AVRInfo
  Support

  ADD_TO_COMPONENT
  AVR
)
