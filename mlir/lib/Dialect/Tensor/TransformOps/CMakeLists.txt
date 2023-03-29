add_mlir_dialect_library(MLIRTensorTransformOps
  TensorTransformOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Tensor/TransformOps

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRTensorDialect
  MLIRTransformDialect
)
