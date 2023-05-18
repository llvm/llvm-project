add_mlir_dialect_library(MLIRTestDynDialect
  TestDynDialect.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  MLIRIR
)
