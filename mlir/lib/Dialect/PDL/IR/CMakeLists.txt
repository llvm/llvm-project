add_mlir_dialect_library(MLIRPDL
  PDL.cpp
  PDLTypes.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/PDL

  DEPENDS
  MLIRPDLOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
  )
