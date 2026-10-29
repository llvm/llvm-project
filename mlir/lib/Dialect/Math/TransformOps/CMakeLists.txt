add_mlir_dialect_library(MLIRMathTransformOps
  MathTransformOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Math/TransformOps

  DEPENDS
  MLIRMathTransformOpsIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRIR
  MLIRMathDialect
  MLIRMathTransforms
  MLIRTransformDialect
  MLIRVectorDialect
  MLIRX86Dialect
)
