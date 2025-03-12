add_mlir_unittest(MLIRPolynomialTests
  PolynomialMathTest.cpp
)
mlir_target_link_libraries(MLIRPolynomialTests
  PRIVATE
  MLIRIR
  MLIRPolynomialDialect
)
