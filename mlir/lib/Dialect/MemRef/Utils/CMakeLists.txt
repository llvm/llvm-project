add_mlir_dialect_library(MLIRMemRefUtils
  MemRefUtils.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/inlude/mlir/Dialect/MemRefDialect

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSideEffectInterfaces
)

