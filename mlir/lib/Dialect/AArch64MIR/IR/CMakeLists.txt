add_mlir_dialect_library(MLIRAArch64MIRDialect
  AArch64MIRDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/AArch64MIR

  DEPENDS
  MLIRAArch64MIROpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRMIRDialect
  MLIRSideEffectInterfaces
  )
