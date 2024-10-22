add_mlir_unittest(MLIRPolynomialTests
  PolynomialMathTest.cpp
)
target_link_libraries(MLIRPolynomialTests
  PRIVATE
  MLIRIR
  MLIRPolynomialDialect
)
