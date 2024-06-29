add_mlir_dialect_library(MLIRMathTransforms
  AlgebraicSimplification.cpp
  ExpandPatterns.cpp
  LegalizeToF32.cpp
  PolynomialApproximation.cpp
  UpliftToFMA.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Math/Transforms

  DEPENDS
  MLIRMathTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRDialectUtils
  MLIRIR
  MLIRMathDialect
  MLIRSCFDialect
  MLIRPass
  MLIRTransforms
  MLIRX86VectorDialect
  MLIRVectorDialect
  )
