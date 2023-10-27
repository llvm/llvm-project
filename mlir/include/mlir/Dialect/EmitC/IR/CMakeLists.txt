add_mlir_dialect(EmitC emitc)
add_mlir_doc(EmitC EmitC Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS EmitCAttributes.td)
mlir_tablegen(EmitCEnums.h.inc -gen-enum-decls)
mlir_tablegen(EmitCEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(EmitCAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(EmitCAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIREmitCAttributesIncGen)
