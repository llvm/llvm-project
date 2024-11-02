add_mlir_dialect_library(MLIRBufferizationTransformOps
  BufferizationTransformOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Bufferization/TransformOps

  DEPENDS
  MLIRBufferizationTransformOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRParser
  MLIRPDLDialect
  MLIRSideEffectInterfaces
  MLIRTransformDialect
  )
