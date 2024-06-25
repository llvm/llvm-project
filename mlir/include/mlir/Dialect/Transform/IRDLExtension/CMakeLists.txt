set(LLVM_TARGET_DEFINITIONS IRDLExtensionOps.td)
mlir_tablegen(IRDLExtensionOps.h.inc -gen-op-decls)
mlir_tablegen(IRDLExtensionOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRTransformDialectIRDLExtensionOpsIncGen)

add_mlir_doc(IRDLExtensionOps IRDLExtensionOps Dialects/ -gen-op-doc)
