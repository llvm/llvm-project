add_mlir_dialect_library(MLIRSparseTensorTransforms
  SparseTensorLowering.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/SparseTensor

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMIR
  MLIRPass
  MLIRStandard
  MLIRSparseTensor
  MLIRTransforms
)
