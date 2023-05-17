set(LLVM_TARGET_DEFINITIONS MemRefTransformOps.td)
mlir_tablegen(MemRefTransformOps.h.inc -gen-op-decls)
mlir_tablegen(MemRefTransformOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRMemRefTransformOpsIncGen)

add_mlir_doc(MemRefTransformOps MemRefTransformOps Dialects/ -gen-op-doc)
