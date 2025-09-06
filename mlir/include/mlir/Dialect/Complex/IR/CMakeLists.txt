add_mlir_dialect(ComplexOps complex)
add_mlir_doc(ComplexOps ComplexOps Dialects/ -gen-dialect-doc  -dialect=complex)

set(LLVM_TARGET_DEFINITIONS ComplexAttributes.td)
mlir_tablegen(ComplexEnums.h.inc -gen-enum-decls)
mlir_tablegen(ComplexEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(ComplexAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(ComplexAttributes.cpp.inc -gen-attrdef-defs)
add_mlir_dialect_tablegen_target(MLIRComplexAttributesIncGen)
