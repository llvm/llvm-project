set(LLVM_TARGET_DEFINITIONS BufferizationTransformOps.td)
mlir_tablegen(BufferizationTransformOps.h.inc -gen-op-decls)
mlir_tablegen(BufferizationTransformOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRBufferizationTransformOpsIncGen)

add_mlir_doc(BufferizationTransformOps BufferizationTransformOps Dialects/ -gen-op-doc)
