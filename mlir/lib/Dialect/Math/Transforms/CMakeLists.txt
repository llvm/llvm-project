add_mlir_dialect_library(MLIRMathTransforms
  AlgebraicSimplification.cpp
  ExpandPatterns.cpp
  PolynomialApproximation.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Math/Transforms

  LINK_LIBS PUBLIC
  MLIRArithmeticDialect
  MLIRDialectUtils  
  MLIRIR
  MLIRMathDialect
  MLIRPass
  MLIRTransforms
  MLIRX86VectorDialect
  MLIRVectorDialect
  MLIRVectorUtils
  )
