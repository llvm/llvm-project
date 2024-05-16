add_mlir_dialect_library(MLIRLibCDialect
  LibCDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/LibC

  DEPENDS
  MLIRLibCIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRFuncDialect
  MLIRSideEffectInterfaces
  )
