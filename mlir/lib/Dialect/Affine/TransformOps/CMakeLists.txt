add_mlir_dialect_library(MLIRAffineTransformOps
  AffineTransformOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Affine/TransformOps

  DEPENDS
  MLIRAffineTransformOpsIncGen

  LINK_LIBS PUBLIC
  MLIRAffineAnalysis
  MLIRAffineDialect
  MLIRFuncDialect
  MLIRIR
  MLIRPDLDialect
  MLIRAffineDialect
  MLIRAffineTransforms
  MLIRAffineUtils
  MLIRTransformDialect
  MLIRVectorDialect
)
