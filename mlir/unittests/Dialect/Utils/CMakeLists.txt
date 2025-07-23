add_mlir_unittest(MLIRDialectUtilsTests
  StructuredOpsUtilsTest.cpp
  ReshapeOpsUtilsTest.cpp
  IndexingUtilsTest.cpp
)
mlir_target_link_libraries(MLIRDialectUtilsTests
  PRIVATE
  MLIRDialectUtils)
