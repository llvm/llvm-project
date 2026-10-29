add_mlir_unittest(MLIROpenACCStandaloneTests
  OpenACCStandaloneTest.cpp
)
mlir_target_link_libraries(MLIROpenACCStandaloneTests
  PRIVATE
  MLIROpenACCDialect
)
