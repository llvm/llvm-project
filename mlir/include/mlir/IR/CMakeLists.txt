add_mlir_interface(OpAsmInterface)
add_mlir_interface(SymbolInterfaces)
add_mlir_interface(RegionKindInterface)

set(LLVM_TARGET_DEFINITIONS BuiltinDialect.td)
mlir_tablegen(BuiltinDialect.h.inc -gen-dialect-decls)
add_public_tablegen_target(MLIRBuiltinDialectIncGen)

set(LLVM_TARGET_DEFINITIONS BuiltinOps.td)
mlir_tablegen(BuiltinOps.h.inc -gen-op-decls)
mlir_tablegen(BuiltinOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRBuiltinOpsIncGen)

set(LLVM_TARGET_DEFINITIONS BuiltinTypes.td)
mlir_tablegen(BuiltinTypes.h.inc -gen-typedef-decls)
mlir_tablegen(BuiltinTypes.cpp.inc -gen-typedef-defs)
add_public_tablegen_target(MLIRBuiltinTypesIncGen)

add_mlir_doc(BuiltinOps -gen-op-doc Builtin Dialects/)
