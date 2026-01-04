add_mlir_dialect_library(MLIROpenACCAnalysis
  OpenACCSupport.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/OpenACC

  LINK_LIBS PUBLIC
  MLIRIR
  MLIROpenACCDialect
  MLIROpenACCUtils
  MLIRSupport
)

