add_llvm_component_library(LLVMMyCFG
  MyCFG.cpp

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  Core
  Support
  )
