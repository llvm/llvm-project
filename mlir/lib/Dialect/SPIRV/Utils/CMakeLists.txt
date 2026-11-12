add_mlir_dialect_library(MLIRSPIRVUtils
  LayoutUtils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/SPIRV

  LINK_LIBS PUBLIC
  MLIRSPIRVDialect
)

