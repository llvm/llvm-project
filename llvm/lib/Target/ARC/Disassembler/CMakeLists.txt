add_llvm_component_library(LLVMARCDisassembler
  ARCDisassembler.cpp

  LINK_COMPONENTS
  ARCInfo
  MCDisassembler
  Support

  ADD_TO_COMPONENT
  ARC
  )
