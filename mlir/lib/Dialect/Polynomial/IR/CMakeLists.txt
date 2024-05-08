add_mlir_dialect_library(MLIRPolynomialDialect
  Polynomial.cpp
  PolynomialAttributes.cpp
  PolynomialDialect.cpp
  PolynomialOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Polynomial

  DEPENDS
  MLIRPolynomialIncGen
  MLIRPolynomialAttributesIncGen
  MLIRBuiltinAttributesIncGen

  LINK_LIBS PUBLIC
  MLIRSupport
  MLIRDialect
  MLIRIR
  MLIRInferTypeOpInterface
  )
