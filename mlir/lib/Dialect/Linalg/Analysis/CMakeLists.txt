add_mlir_dialect_library(MLIRLinalgAnalysis
  DependenceAnalysis.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Linalg

  LINK_LIBS PUBLIC
  MLIRAffineAnalysis
  MLIRAnalysis
  MLIRIR
  MLIRLinalgDialect
  MLIRMemRefDialect
  )
