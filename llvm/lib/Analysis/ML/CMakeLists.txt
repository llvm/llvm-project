add_llvm_component_library(LLVMMLPolicies
  InlineFeaturesAnalysis.cpp

  DEPENDS
  intrinsics_gen
  )
