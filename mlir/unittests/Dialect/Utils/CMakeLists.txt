add_mlir_unittest(MLIRDialectUtilsTests
  StructuredOpsUtilsTest.cpp
  IndexingUtilsTest.cpp
)
target_link_libraries(MLIRDialectUtilsTests
  PRIVATE
  MLIRDialectUtils)
