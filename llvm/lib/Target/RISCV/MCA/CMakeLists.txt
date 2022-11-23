add_llvm_component_library(LLVMRISCVTargetMCA
  RISCVCustomBehaviour.cpp

  LINK_COMPONENTS
  MC
  MCParser
  RISCVDesc
  RISCVInfo
  Support
  MCA

  ADD_TO_COMPONENT
  RISCV
  )
