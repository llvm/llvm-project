add_llvm_component_library(LLVMMCJIT
  MCJIT.cpp

  DEPENDS
  intrinsics_gen
  )
