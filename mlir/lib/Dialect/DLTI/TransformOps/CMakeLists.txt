add_mlir_dialect_library(MLIRDLTITransformOps
  DLTITransformOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/DLTI/TransformOps

  DEPENDS
  MLIRDLTITransformOpsIncGen
  MLIRDLTIDialect

  LINK_LIBS PUBLIC
  MLIRDLTIDialect
  MLIRSideEffectInterfaces
  MLIRTransformDialect
  )
