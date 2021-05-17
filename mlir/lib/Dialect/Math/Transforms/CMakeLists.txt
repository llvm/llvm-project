add_mlir_dialect_library(MLIRMathTransforms
  ExpandTanh.cpp
  PolynomialApproximation.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Math/Transforms

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMIR
  MLIRMath
  MLIRPass
  MLIRStandard
  MLIRTransforms
  )
