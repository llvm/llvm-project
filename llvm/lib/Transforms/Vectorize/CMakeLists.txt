add_llvm_component_library(LLVMVectorize
  LoadStoreVectorizer.cpp
  LoopIdiomVectorize.cpp
  LoopVectorizationLegality.cpp
  LoopVectorize.cpp
  SandboxVectorizer/DependencyGraph.cpp
  SandboxVectorizer/Passes/BottomUpVec.cpp
  SandboxVectorizer/Region.cpp
  SandboxVectorizer/SandboxVectorizer.cpp
  SLPVectorizer.cpp
  Vectorize.cpp
  VectorCombine.cpp
  VPlan.cpp
  VPlanAnalysis.cpp
  VPlanHCFGBuilder.cpp
  VPlanRecipes.cpp
  VPlanSLP.cpp
  VPlanTransforms.cpp
  VPlanUnroll.cpp
  VPlanVerifier.cpp
  VPlanUtils.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Transforms
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Transforms/Vectorize
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Transforms/Vectorize/SandboxVectorizer

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  Analysis
  Core
  Support
  TransformUtils
  SandboxIR
  )
