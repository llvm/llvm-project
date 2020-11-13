add_llvm_component_library(LLVMBPFDisassembler
  BPFDisassembler.cpp

  LINK_COMPONENTS
  MCDisassembler
  BPFInfo
  Support

  ADD_TO_COMPONENT
  BPF
)

