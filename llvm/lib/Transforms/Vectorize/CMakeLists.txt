set(LLVM_TABLEGEN_PROJECT LLVM_HEADERS)
set(LLVM_TARGET_DEFINITIONS VectorizeOptions.td)
tablegen(LLVM VectorizeOptions.inc -gen-opt-parser-defs)
add_public_tablegen_target(VectorizeOptionsTableGen)

add_llvm_component_library(LLVMVectorize
  LoadStoreVectorizer.cpp
  LoopIdiomVectorize.cpp
  LoopVectorizationLegality.cpp
  LoopVectorizationPlanner.cpp
  LoopVectorize.cpp
  SandboxVectorizer/DependencyGraph.cpp
  SandboxVectorizer/InstrMaps.cpp
  SandboxVectorizer/Interval.cpp
  SandboxVectorizer/Legality.cpp
  SandboxVectorizer/Passes/BundleVec.cpp
  SandboxVectorizer/Passes/LoadStoreVec.cpp
  SandboxVectorizer/Passes/PackReuse.cpp
  SandboxVectorizer/Passes/RegionsFromBBs.cpp
  SandboxVectorizer/Passes/RegionsFromMetadata.cpp
  SandboxVectorizer/Passes/SeedCollection.cpp
  SandboxVectorizer/Passes/TransactionAcceptOrRevert.cpp
  SandboxVectorizer/Passes/TransactionSave.cpp
  SandboxVectorizer/RegionWithScore.cpp
  SandboxVectorizer/SandboxVectorizer.cpp
  SandboxVectorizer/SandboxVectorizerIR.cpp
  SandboxVectorizer/SandboxVectorizerPassBuilder.cpp
  SandboxVectorizer/Scheduler.cpp
  SandboxVectorizer/SeedCollector.cpp
  SandboxVectorizer/VecUtils.cpp
  SLPVectorizer/SLPCompatibilityAnalysis.cpp
  SLPVectorizer/SLPCostAnalysis.cpp
  SLPVectorizer/SLPMemoryUtils.cpp
  SLPVectorizer/SLPReductionUtils.cpp
  SLPVectorizer/SLPTypeUtils.cpp
  SLPVectorizer/SLPUtils.cpp
  SLPVectorizer.cpp
  Vectorize.cpp
  VectorCombine.cpp
  VectorizeOptions.cpp
  VPlan.cpp
  VPlanAnalysis.cpp
  VPlanConstruction.cpp
  VPlanDominatorTree.cpp
  VPlanEVLTailFolding.cpp
  VPlanLowering.cpp
  VPlanPredicator.cpp
  VPlanRecipes.cpp
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
  VectorizeOptionsTableGen

  LINK_COMPONENTS
  Analysis
  Core
  Option
  Support
  TransformUtils
  SandboxIR
  )
