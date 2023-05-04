add_llvm_component_library(LLVMAVRAsmParser
  AVRAsmParser.cpp

  LINK_COMPONENTS
  AVRDesc
  AVRInfo
  CodeGen
  MC
  MCParser
  Support

  ADD_TO_COMPONENT
  AVR
  )
