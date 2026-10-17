add_mlir_dialect_library(MLIRX86Utils
  X86Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/X86/Utils

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRDialectUtils
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRVectorDialect
  )
