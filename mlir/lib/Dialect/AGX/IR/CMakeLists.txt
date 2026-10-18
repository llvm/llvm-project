add_mlir_dialect_library(MLIRAGXDialect
  AGXDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/AGX

  DEPENDS
  MLIRAGXEnumsGen
  MLIRAGXIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSideEffectInterfaces
  )
