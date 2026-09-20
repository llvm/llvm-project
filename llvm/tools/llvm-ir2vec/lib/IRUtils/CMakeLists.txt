add_llvm_library(LLVMIREmbUtils STATIC
  IRUtils.cpp

  BUILDTREE_ONLY

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  Analysis
  Core
  Support
)
