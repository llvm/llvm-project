add_llvm_component_library(LLVMRISCVAsmParser
  RISCVAsmParser.cpp

  LINK_COMPONENTS
  MC
  MCParser
  RISCVDesc
  RISCVInfo
  Support

  ADD_TO_COMPONENT
  RISCV
  )
