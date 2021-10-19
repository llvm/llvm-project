add_mlir_unittest(MLIRDialectUtilsTests
  StructuredOpsUtilsTest.cpp
)
target_link_libraries(MLIRDialectUtilsTests
  PRIVATE
  MLIRDialectUtils)
