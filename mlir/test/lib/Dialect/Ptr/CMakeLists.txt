add_mlir_library(MLIRPtrTestPasses
  TestTypeOffset.cpp

  EXCLUDE_FROM_LIBMLIR
)
mlir_target_link_libraries(MLIRPtrTestPasses PUBLIC
  MLIRDataLayoutInterfaces
  MLIRIR
  MLIRPass
  MLIRPtrDialect
)
