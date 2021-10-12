add_mlir_library(MLIRTransformUtils
  DialectConversion.cpp
  FoldUtils.cpp
  GreedyPatternRewriteDriver.cpp
  InliningUtils.cpp
  LoopFusionUtils.cpp
  LoopUtils.cpp
  RegionUtils.cpp
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Transforms

  DEPENDS
  MLIRStandardOpsIncGen

  LINK_LIBS PUBLIC
  MLIRAffine
  MLIRArithmetic
  MLIRAnalysis
  MLIRLoopAnalysis
  MLIRMemRef
  MLIRSCF
  MLIRPass
  MLIRRewrite
  MLIRStandard
  )
