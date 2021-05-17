# Exclude tests from libMLIR.so
add_mlir_library(MLIRMathTestPasses
  TestExpandTanh.cpp
  TestPolynomialApproximation.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  MLIRMathTransforms
  MLIRPass
  MLIRTransformUtils
  MLIRVector
  )
