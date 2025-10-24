add_llvm_component_library(LLVMDTLTO
  DTLTO.cpp

  LINK_COMPONENTS
  Core
  Support
  )
