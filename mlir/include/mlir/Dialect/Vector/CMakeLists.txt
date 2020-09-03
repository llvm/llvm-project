add_mlir_dialect(VectorOps vector)
add_mlir_doc(VectorOps -gen-op-doc VectorOps Dialects/)

set(LLVM_TARGET_DEFINITIONS VectorTransformPatterns.td)
mlir_tablegen(VectorTransformPatterns.h.inc -gen-rewriters)
add_public_tablegen_target(MLIRVectorTransformPatternsIncGen)
