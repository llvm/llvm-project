add_mlir_dialect_library(MLIRSparseTensorUtils
  Merger.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/SparseTensor

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLinalg
)
