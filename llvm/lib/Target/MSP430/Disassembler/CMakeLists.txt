add_llvm_component_library(LLVMMSP430Disassembler
  MSP430Disassembler.cpp

  LINK_COMPONENTS
  MCDisassembler
  MSP430Info
  Support

  ADD_TO_COMPONENT
  MSP430
  )
