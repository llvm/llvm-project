add_mlir_dialect_library(MLIRReducerDialect
  ReducerOps.cpp

  LINK_LIBS PUBLIC
  MLIRDialectUtils
  MLIRIR
  )
