add_mlir_dialect_library(MLIRTensorTransformOps
  TensorTransformOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Tensor/TransformOps

  DEPENDS
  MLIRTensorTransformOpsIncGen

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRIR
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRTensorTransforms
  MLIRTensorUtils
  MLIRTransformDialect
)
