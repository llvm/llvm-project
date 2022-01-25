add_mlir_dialect_library(MLIRMathTransforms
  AlgebraicSimplification.cpp
  ExpandTanh.cpp
  PolynomialApproximation.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Math/Transforms

  LINK_LIBS PUBLIC
  MLIRArithmetic
  MLIRIR
  MLIRMath
  MLIRPass
  MLIRStandard
  MLIRTransforms
  MLIRX86Vector
  MLIRVector
  )
