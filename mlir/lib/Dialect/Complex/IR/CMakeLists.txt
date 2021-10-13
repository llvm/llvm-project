add_mlir_dialect_library(MLIRComplex
  ComplexOps.cpp
  ComplexDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Complex

  DEPENDS
  MLIRComplexOpsIncGen

  LINK_LIBS PUBLIC
  MLIRArithmetic
  MLIRDialect
  MLIRIR
  MLIRStandard
  )
