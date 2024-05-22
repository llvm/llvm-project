add_mlir_dialect_library(MLIRMathDialect
  MathOps.cpp
  MathDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Math

  DEPENDS
  MLIRMathOpsIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRDialect
  MLIRIR
  MLIRUBDialect
  )
