add_mlir_unittest(MLIRTransformDialectTests
  BuildOnlyExtensionTest.cpp
)
target_link_libraries(MLIRTransformDialectTests
  PRIVATE
  MLIRFuncDialect
  MLIRTransformDialect
)
