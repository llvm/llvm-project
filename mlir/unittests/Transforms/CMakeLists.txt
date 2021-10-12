add_mlir_unittest(MLIRTransformsTests
  DialectConversion.cpp
)
target_link_libraries(MLIRTransformsTests
  PRIVATE
  MLIRTransforms)
