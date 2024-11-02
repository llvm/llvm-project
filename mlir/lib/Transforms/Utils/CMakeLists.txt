add_mlir_library(MLIRTransformUtils
  CommutativityUtils.cpp
  ControlFlowSinkUtils.cpp
  DialectConversion.cpp
  FoldUtils.cpp
  GreedyPatternRewriteDriver.cpp
  InliningUtils.cpp
  LoopInvariantCodeMotionUtils.cpp
  RegionUtils.cpp
  TopologicalSortUtils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Transforms

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRLoopLikeInterface
  MLIRSideEffectInterfaces
  MLIRRewrite
  )
