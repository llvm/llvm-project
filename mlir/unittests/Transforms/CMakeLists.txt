add_mlir_unittest(MLIRTransformsTests
  Canonicalizer.cpp
  DialectConversion.cpp
  GreedyPatternRewriteDriver.cpp
)
mlir_target_link_libraries(MLIRTransformsTests
  PRIVATE
  MLIRParser
  MLIRTransforms)

target_include_directories(MLIRTransformsTests PRIVATE "${MLIR_BINARY_DIR}/test/lib/Dialect/Test")
target_link_libraries(MLIRTransformsTests PRIVATE MLIRTestDialect)
