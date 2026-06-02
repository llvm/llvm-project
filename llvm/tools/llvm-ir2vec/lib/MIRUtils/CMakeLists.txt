add_llvm_library(LLVMMIREmbUtils STATIC
  MIRUtils.cpp

  BUILDTREE_ONLY

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  CodeGen
  Target
)
