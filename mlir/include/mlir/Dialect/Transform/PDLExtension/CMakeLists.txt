set(LLVM_TARGET_DEFINITIONS PDLExtensionOps.td)
mlir_tablegen(PDLExtensionOps.h.inc -gen-op-decls)
mlir_tablegen(PDLExtensionOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRTransformDialectPDLExtensionOpsIncGen)

add_mlir_doc(PDLExtensionOps PDLExtensionOps Dialects/ -gen-op-doc)
