add_mlir_dialect_library(MLIRArithmeticUtils
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Arithmetic/Utils

  LINK_LIBS PUBLIC
  MLIRArithmetic
  MLIRDialect
  MLIRIR
  )
