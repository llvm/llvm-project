add_llvm_library(LLVMIREmbUtils STATIC
  IRUtils.cpp

  PARTIAL_SOURCES_INTENDED
  BUILDTREE_ONLY

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  Analysis
  Core
  Support
)

add_llvm_library(LLVMMIREmbUtils STATIC
  MIRUtils.cpp

  PARTIAL_SOURCES_INTENDED
  BUILDTREE_ONLY

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  CodeGen
  Target
)
