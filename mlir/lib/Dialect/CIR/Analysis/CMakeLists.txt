add_mlir_dialect_library(MLIRCIRAnalysis
  Lifetime.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/CIR

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRIR
  )

