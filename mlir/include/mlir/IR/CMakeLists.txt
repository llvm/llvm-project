add_mlir_interface(OpAsmInterface)
add_mlir_interface(SymbolInterfaces)
add_mlir_interface(RegionKindInterface)

set(LLVM_TARGET_DEFINITIONS BuiltinOps.td)
mlir_tablegen(BuiltinOps.h.inc -gen-op-decls)
mlir_tablegen(BuiltinOps.cpp.inc -gen-op-defs)
mlir_tablegen(BuiltinDialect.h.inc -gen-dialect-decls)
add_public_tablegen_target(MLIRBuiltinOpsIncGen)

add_mlir_doc(BuiltinOps -gen-op-doc Builtin Dialects/)
