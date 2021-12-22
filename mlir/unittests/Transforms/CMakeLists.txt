add_mlir_unittest(MLIRTransformsTests
  Canonicalizer.cpp
  DialectConversion.cpp
)
target_link_libraries(MLIRTransformsTests
  PRIVATE
  MLIRTransforms)
