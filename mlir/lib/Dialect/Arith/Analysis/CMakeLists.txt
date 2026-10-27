add_mlir_dialect_library(MLIRArithAnalysis
  IntegerDivisibility.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Arith/Analysis

  LINK_LIBS PUBLIC
  MLIRAnalysisDataFlow
  MLIRArithDialect
  MLIRIR
  MLIRLoopLikeInterface
  )
