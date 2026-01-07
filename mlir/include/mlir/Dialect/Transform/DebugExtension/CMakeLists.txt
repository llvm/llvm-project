set(LLVM_TARGET_DEFINITIONS DebugExtensionOps.td)
mlir_tablegen(DebugExtensionOps.h.inc -gen-op-decls)
mlir_tablegen(DebugExtensionOps.cpp.inc -gen-op-defs)
add_mlir_dialect_tablegen_target(MLIRTransformDialectDebugExtensionOpsIncGen)

add_mlir_doc(DebugExtensionOps DebugExtensionOps Dialects/ -gen-op-doc)
