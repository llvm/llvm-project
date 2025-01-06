add_mlir_unittest(MLIROpenACCTests
  OpenACCOpsTest.cpp
)
mlir_target_link_libraries(MLIROpenACCTests
  PRIVATE
  MLIRIR
  MLIROpenACCDialect
)
