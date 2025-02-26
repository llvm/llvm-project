# Exclude tests from libMLIR.so
add_mlir_library(MLIRMathTestPasses
  TestAlgebraicSimplification.cpp
  TestExpandMath.cpp
  TestPolynomialApproximation.cpp

  EXCLUDE_FROM_LIBMLIR
  )
mlir_target_link_libraries(MLIRMathTestPasses PUBLIC
  MLIRMathTransforms
  MLIRPass
  MLIRTransformUtils
  MLIRVectorDialect
  MLIRX86VectorDialect
  )
