add_mlir_library(MLIRCAPIStandard

  StandardDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir-c

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRStandard
  )
