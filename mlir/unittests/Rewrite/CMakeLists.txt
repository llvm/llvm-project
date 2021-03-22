add_mlir_unittest(MLIRRewriteTests
  PatternBenefit.cpp
)
target_link_libraries(MLIRRewriteTests
  PRIVATE
  MLIRRewrite
  MLIRTransformUtils)
