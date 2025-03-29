add_mlir_unittest(MLIRTransformsTests
  Canonicalizer.cpp
  DialectConversion.cpp
)
mlir_target_link_libraries(MLIRTransformsTests
  PRIVATE
  MLIRParser
  MLIRTransforms)
