add_llvm_component_library(LLVMRISCVUtils
  RISCVBaseInfo.cpp
  RISCVMatInt.cpp

  LINK_COMPONENTS
  Support

  ADD_TO_COMPONENT
  RISCV
  )
