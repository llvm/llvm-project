add_mlir_dialect_library(MLIRX86VectorUtils
  X86VectorUtils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/X86Vector/Utils

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRDialectUtils
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRVectorDialect
  )
