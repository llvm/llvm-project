add_llvm_component_library(LLVMVEDisassembler
  VEDisassembler.cpp

  LINK_COMPONENTS
  MCDisassembler
  VEInfo
  Support

  ADD_TO_COMPONENT
  VE
  )
