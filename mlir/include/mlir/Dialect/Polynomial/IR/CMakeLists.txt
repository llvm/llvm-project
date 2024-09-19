add_mlir_dialect(Polynomial polynomial)
add_mlir_doc(Polynomial PolynomialDialect Dialects/ -gen-dialect-doc -dialect=polynomial)

set(LLVM_TARGET_DEFINITIONS PolynomialAttributes.td)
mlir_tablegen(PolynomialAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=polynomial)
mlir_tablegen(PolynomialAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=polynomial)
add_public_tablegen_target(MLIRPolynomialAttributesIncGen)
