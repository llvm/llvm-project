add_mlir_dialect_library(MLIROpenACCDialect
  IR/OpenACC.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/OpenACC

  DEPENDS
  MLIROpenACCOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  )

