add_mlir_library(MLIRTransformUtils
  CFGToSCF.cpp
  CommutativityUtils.cpp
  ControlFlowSinkUtils.cpp
  DialectConversion.cpp
  FoldUtils.cpp
  GreedyPatternRewriteDriver.cpp
  Inliner.cpp
  InliningUtils.cpp
  LoopInvariantCodeMotionUtils.cpp
  RegionUtils.cpp
  WalkPatternRewriteDriver.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Transforms

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRCallInterfaces
  MLIRControlFlowInterfaces
  MLIRFunctionInterfaces
  MLIRLoopLikeInterface
  MLIRSideEffectInterfaces
  MLIRSubsetOpInterface
  MLIRRewrite
  )

# Anonymous namespace debugging infrastructure and static helper conflicts.
set_source_files_properties(
  GreedyPatternRewriteDriver.cpp
  PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
