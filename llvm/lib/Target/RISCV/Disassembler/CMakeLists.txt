add_llvm_component_library(LLVMRISCVDisassembler
  RISCVDisassembler.cpp

  LINK_COMPONENTS
  MCDisassembler
  RISCVInfo
  Support

  ADD_TO_COMPONENT
  RISCV
  )
