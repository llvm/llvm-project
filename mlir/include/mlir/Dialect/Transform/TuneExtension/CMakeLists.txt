set(LLVM_TARGET_DEFINITIONS TuneExtensionOps.td)
mlir_tablegen(TuneExtensionOps.h.inc -gen-op-decls)
mlir_tablegen(TuneExtensionOps.cpp.inc -gen-op-defs)
add_mlir_dialect_tablegen_target(MLIRTransformDialectTuneExtensionOpsIncGen)

add_mlir_doc(TuneExtensionOps TuneExtensionOps Dialects/ -gen-op-doc)
