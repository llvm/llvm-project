add_mlir_dialect_library(MLIRMemRefTransformOps
  MemRefTransformOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/MemRef/TransformOps

  DEPENDS
  MLIRMemRefTransformOpsIncGen

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRArithDialect
  MLIRIR
  MLIRPDLDialect
  MLIRMemRefDialect
  MLIRMemRefTransforms
  MLIRTransformDialect
)
