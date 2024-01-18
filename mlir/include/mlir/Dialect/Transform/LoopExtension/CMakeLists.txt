set(LLVM_TARGET_DEFINITIONS LoopExtensionOps.td)
mlir_tablegen(LoopExtensionOps.h.inc -gen-op-decls)
mlir_tablegen(LoopExtensionOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRTransformDialectLoopExtensionOpsIncGen)

add_mlir_doc(LoopExtensionOps LoopExtensionOps Dialects/ -gen-op-doc)
