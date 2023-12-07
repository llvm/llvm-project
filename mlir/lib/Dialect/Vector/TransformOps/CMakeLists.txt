add_mlir_dialect_library(MLIRVectorTransformOps
  VectorTransformOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Vector/TransformOps

  DEPENDS
  MLIRVectorTransformOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRVectorDialect
  MLIRVectorToLLVM
  MLIRVectorTransforms
  MLIRSideEffectInterfaces
  MLIRTransformDialect
  MLIRVectorDialect
  MLIRVectorToSCF
  MLIRX86VectorTransforms
  )
