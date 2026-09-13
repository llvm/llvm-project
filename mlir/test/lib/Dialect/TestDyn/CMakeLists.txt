add_mlir_library(MLIRTestDynDialect
  TestDynDialect.cpp

  EXCLUDE_FROM_LIBMLIR

  DEPENDS
  mlir-headers
)
mlir_target_link_libraries(MLIRTestDynDialect PUBLIC
  MLIRIR
)
