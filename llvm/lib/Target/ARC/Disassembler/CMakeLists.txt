add_llvm_component_library(LLVMARCDisassembler
  ARCDisassembler.cpp

  LINK_COMPONENTS
  MCDisassembler
  Support
  ARCInfo

  ADD_TO_COMPONENT
  ARC
  )
