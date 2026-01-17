add_llvm_library(emb_utils STATIC
  utils.cpp

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  Analysis
  CodeGen
  Core
  Support
  Target
)
