add_llvm_library(LLVMEmbUtils STATIC
  Utils.cpp

  BUILDTREE_ONLY

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  Analysis
  CodeGen
  Core
  Support
  Target
)
