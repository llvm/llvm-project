add_mlir_dialect(ComplexOps complex)
add_mlir_doc(ComplexOps ComplexOps Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS ComplexAttributes.td)
mlir_tablegen(ComplexAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(ComplexAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRComplexAttributesIncGen)
