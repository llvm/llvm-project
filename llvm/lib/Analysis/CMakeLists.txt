if (DEFINED LLVM_HAVE_TF_AOT OR DEFINED LLVM_HAVE_TF_API)
  include(TensorFlowCompile)
  set(LLVM_INLINER_MODEL_PATH_DEFAULT "models/inliner-Oz")

  # This url points to the most recent most which is known to be compatible with
  # LLVM. When better models are published, this url should be updated to aid
  # discoverability.
  set(LLVM_INLINER_MODEL_CURRENT_URL "https://github.com/google/ml-compiler-opt/releases/download/inlining-Oz-v0.1/inlining-Oz-acabaf6-v0.1.tar.gz")

  if (DEFINED LLVM_HAVE_TF_AOT)
    # If the path is empty, autogenerate the model
    if (NOT DEFINED LLVM_INLINER_MODEL_PATH OR "${LLVM_INLINER_MODEL_PATH}" STREQUAL "")
      set(LLVM_INLINER_MODEL_PATH "autogenerate")
      message(WARNING "LLVM_INLINER_MODEL_PATH was not set: autogenerating a model to finish the build.")
    endif()

    tf_find_and_compile(
      ${LLVM_INLINER_MODEL_PATH}
      ${LLVM_INLINER_MODEL_CURRENT_URL}
      ${LLVM_INLINER_MODEL_PATH_DEFAULT}
      "models/inlining/config.py"
      serve
      action
      InlinerSizeModel
      llvm::InlinerSizeModel
    )
  endif()

  if (DEFINED LLVM_HAVE_TF_API)
    list(APPEND MLLinkDeps ${tensorflow_c_api})
  endif()
endif()

add_llvm_component_library(LLVMAnalysis
  AliasAnalysis.cpp
  AliasAnalysisEvaluator.cpp
  AliasAnalysisSummary.cpp
  AliasSetTracker.cpp
  Analysis.cpp
  AssumeBundleQueries.cpp
  AssumptionCache.cpp
  BasicAliasAnalysis.cpp
  BlockFrequencyInfo.cpp
  BlockFrequencyInfoImpl.cpp
  BranchProbabilityInfo.cpp
  CFG.cpp
  CFGPrinter.cpp
  CFLAndersAliasAnalysis.cpp
  CFLSteensAliasAnalysis.cpp
  CGSCCPassManager.cpp
  CallGraph.cpp
  CallGraphSCCPass.cpp
  CallPrinter.cpp
  CaptureTracking.cpp
  CmpInstAnalysis.cpp
  CostModel.cpp
  CodeMetrics.cpp
  ConstantFolding.cpp
  DDG.cpp
  DDGPrinter.cpp
  ConstraintSystem.cpp
  Delinearization.cpp
  DemandedBits.cpp
  DependenceAnalysis.cpp
  DependenceGraphBuilder.cpp
  DevelopmentModeInlineAdvisor.cpp
  DivergenceAnalysis.cpp
  DomPrinter.cpp
  DomTreeUpdater.cpp
  DominanceFrontier.cpp
  EHPersonalities.cpp
  FunctionPropertiesAnalysis.cpp
  GlobalsModRef.cpp
  GuardUtils.cpp
  HeatUtils.cpp
  IRSimilarityIdentifier.cpp
  IVDescriptors.cpp
  IVUsers.cpp
  ImportedFunctionsInliningStatistics.cpp
  IndirectCallPromotionAnalysis.cpp
  InlineCost.cpp
  InlineAdvisor.cpp
  InlineSizeEstimatorAnalysis.cpp
  InstCount.cpp
  InstructionPrecedenceTracking.cpp
  InstructionSimplify.cpp
  Interval.cpp
  IntervalPartition.cpp
  LazyBranchProbabilityInfo.cpp
  LazyBlockFrequencyInfo.cpp
  LazyCallGraph.cpp
  LazyValueInfo.cpp
  LegacyDivergenceAnalysis.cpp
  Lint.cpp
  Loads.cpp
  LoopAccessAnalysis.cpp
  LoopAnalysisManager.cpp
  LoopCacheAnalysis.cpp
  LoopNestAnalysis.cpp
  LoopUnrollAnalyzer.cpp
  LoopInfo.cpp
  LoopPass.cpp
  MLInlineAdvisor.cpp
  MemDepPrinter.cpp
  MemDerefPrinter.cpp
  MemoryBuiltins.cpp
  MemoryDependenceAnalysis.cpp
  MemoryLocation.cpp
  MemorySSA.cpp
  MemorySSAUpdater.cpp
  ModuleDebugInfoPrinter.cpp
  ModuleSummaryAnalysis.cpp
  MustExecute.cpp
  ObjCARCAliasAnalysis.cpp
  ObjCARCAnalysisUtils.cpp
  ObjCARCInstKind.cpp
  OptimizationRemarkEmitter.cpp
  OverflowInstAnalysis.cpp
  PHITransAddr.cpp
  PhiValues.cpp
  PostDominators.cpp
  ProfileSummaryInfo.cpp
  PtrUseVisitor.cpp
  RegionInfo.cpp
  RegionPass.cpp
  RegionPrinter.cpp
  ReleaseModeModelRunner.cpp
  ReplayInlineAdvisor.cpp
  ScalarEvolution.cpp
  ScalarEvolutionAliasAnalysis.cpp
  ScalarEvolutionDivision.cpp
  ScalarEvolutionNormalization.cpp
  StackLifetime.cpp
  StackSafetyAnalysis.cpp
  SyncDependenceAnalysis.cpp
  SyntheticCountsUtils.cpp
  TFUtils.cpp
  TargetLibraryInfo.cpp
  TargetTransformInfo.cpp
  Trace.cpp
  TypeBasedAliasAnalysis.cpp
  TypeMetadataUtils.cpp
  ScopedNoAliasAA.cpp
  ValueLattice.cpp
  ValueLatticeUtils.cpp
  ValueTracking.cpp
  VectorUtils.cpp
  VFABIDemangling.cpp
  ${GeneratedMLSources}

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Analysis

  DEPENDS
  intrinsics_gen
  ${MLDeps}

  LINK_LIBS
  ${MLLinkDeps}

  LINK_COMPONENTS
  BinaryFormat
  Core
  Object
  ProfileData
  Support
  )
