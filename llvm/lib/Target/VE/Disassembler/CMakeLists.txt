add_llvm_component_library(LLVMVEDisassembler
  VEDisassembler.cpp

  LINK_COMPONENTS
  MCDisassembler
  MC
  VEInfo
  Support

  ADD_TO_COMPONENT
  VE
  )
