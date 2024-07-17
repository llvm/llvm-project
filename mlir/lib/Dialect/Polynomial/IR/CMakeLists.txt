set(LLVM_TARGET_DEFINITIONS PolynomialCanonicalization.td)
mlir_tablegen(PolynomialCanonicalization.inc -gen-rewriters)
add_public_tablegen_target(MLIRPolynomialCanonicalizationIncGen)

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
  MLIRPolynomialCanonicalizationIncGen
  MLIRBuiltinAttributesIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRSupport
  MLIRDialect
  MLIRIR
  MLIRInferTypeOpInterface
  )
