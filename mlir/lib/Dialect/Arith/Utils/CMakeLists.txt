add_mlir_dialect_library(MLIRArithUtils
  NarrowFloatExpansion.cpp
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Arith/Utils

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRAsmParser
  MLIRComplexDialect
  MLIRDialect
  MLIRDialectUtils
  MLIRIR
  )
