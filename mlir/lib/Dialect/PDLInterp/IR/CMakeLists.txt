add_mlir_dialect_library(MLIRPDLInterpDialect
  PDLInterp.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/PDLInterp

  DEPENDS
  MLIRPDLInterpOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPDLDialect
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
  )
