add_mlir_dialect_library(MLIRPartTensorPipelines
  PartTensorPipelines.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/PartTensor

  LINK_LIBS PUBLIC
  MLIRSparseTensorPipelines
  MLIRPartTensor
  MLIRPartTensorTransforms
)
