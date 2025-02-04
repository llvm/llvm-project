add_mlir_unittest(MLIRRewriteTests
  PatternBenefit.cpp
)
mlir_target_link_libraries(MLIRRewriteTests
  PRIVATE
  MLIRRewrite
  MLIRTransformUtils)
