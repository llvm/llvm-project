set(LLVM_TARGET_DEFINITIONS SMTExtensionOps.td)
mlir_tablegen(SMTExtensionOps.h.inc -gen-op-decls)
mlir_tablegen(SMTExtensionOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRTransformDialectSMTExtensionOpsIncGen)

add_mlir_doc(SMTExtensionOps SMTExtensionOps Dialects/ -gen-op-doc)
