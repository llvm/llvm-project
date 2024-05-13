set(LLVM_TARGET_DEFINITIONS VectorTransformOps.td)
mlir_tablegen(VectorTransformOps.h.inc -gen-op-decls)
mlir_tablegen(VectorTransformOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRVectorTransformOpsIncGen)

add_mlir_doc(VectorTransformOps VectorTransformOps Dialects/ -gen-op-doc)
