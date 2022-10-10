add_mlir_dialect_library(MLIRSparseTensorDialect
  SparseTensorDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/SparseTensor

  DEPENDS
  MLIRSparseTensorAttrDefsIncGen
  MLIRSparseTensorOpsIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRDialect
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRSupport
  )
