add_mlir_dialect(EmitC emitc)
add_mlir_doc(EmitC EmitC Dialects/ -gen-dialect-doc -dialect emitc)

set(LLVM_TARGET_DEFINITIONS EmitCInterfaces.td)
mlir_tablegen(EmitCInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(EmitCInterfaces.cpp.inc -gen-op-interface-defs)
add_mlir_generic_tablegen_target(MLIREmitCInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS EmitCAttributes.td)
mlir_tablegen(EmitCEnums.h.inc -gen-enum-decls)
mlir_tablegen(EmitCEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(EmitCAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(EmitCAttributes.cpp.inc -gen-attrdef-defs)
add_mlir_dialect_tablegen_target(MLIREmitCAttributesIncGen)
