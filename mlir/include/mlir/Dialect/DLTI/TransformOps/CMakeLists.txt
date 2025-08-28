set(LLVM_TARGET_DEFINITIONS DLTITransformOps.td)
mlir_tablegen(DLTITransformOps.h.inc -gen-op-decls)
mlir_tablegen(DLTITransformOps.cpp.inc -gen-op-defs)
add_mlir_dialect_tablegen_target(MLIRDLTITransformOpsIncGen)

add_mlir_doc(DLTITransformOps DLTITransformOps Dialects/ -gen-op-doc)
