add_mlir_dialect_library(MLIRSparseTensorTransformOps
  SparseTensorTransformOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/SparseTensor/TransformOps

  DEPENDS
  MLIRSparseTensorTransformOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSparseTensorDialect
  MLIRParser
  MLIRSideEffectInterfaces
  MLIRTransformDialect
  MLIRTransformDialectUtils
  )
