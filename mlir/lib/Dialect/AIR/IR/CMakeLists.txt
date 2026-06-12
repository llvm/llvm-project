add_mlir_dialect_library(MLIRAIRDialect
  AIRDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/AIR

  DEPENDS
  MLIRAIROpsIncGen
  MLIRAIRAttributesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRGPUDialect
  )
