set(BUILD_SHARED_LIBS OFF)

add_llvm_internal_library(LLVMMLPolicies PUBLIC
  InlineFeaturesAnalysis.cpp

  DEPENDS
  intrinsics_gen
  )
