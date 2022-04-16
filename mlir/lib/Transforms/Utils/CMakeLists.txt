add_mlir_library(MLIRTransformUtils
  ControlFlowSinkUtils.cpp
  DialectConversion.cpp
  FoldUtils.cpp
  GreedyPatternRewriteDriver.cpp
  InliningUtils.cpp
  LoopInvariantCodeMotionUtils.cpp
  RegionUtils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Transforms

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRRewrite
  )
