add_mlir_unittest(MLIROpenACCTests
  OpenACCOpsTest.cpp
)
target_link_libraries(MLIROpenACCTests
  PRIVATE
  MLIRIR
  MLIROpenACCDialect
)
