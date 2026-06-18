add_mlir_dialect_library(MLIRFuncUtils
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Func/Utils

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRDialect
  MLIRDialectUtils
  MLIRIR
  )
