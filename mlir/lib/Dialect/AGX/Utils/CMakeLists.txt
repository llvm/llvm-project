add_mlir_dialect_library(MLIRAGXUtils
  ChipInfo.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/AGX/Utils

  LINK_LIBS PUBLIC
  MLIRSupport
  )
