add_llvm_component_library(LLVMVectorize
  LoadStoreVectorizer.cpp
  LoopVectorizationLegality.cpp
  LoopVectorize.cpp
  SLPVectorizer.cpp
  Vectorize.cpp
  VPlan.cpp
  VPlanHCFGBuilder.cpp
  VPlanPredicator.cpp
  VPlanSLP.cpp
  VPlanTransforms.cpp
  VPlanVerifier.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Transforms

  DEPENDS
  intrinsics_gen
  )
