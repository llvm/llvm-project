set(LLVM_TARGET_DEFINITIONS LinalgTransformOps.td)
mlir_tablegen(LinalgTransformOps.h.inc -gen-op-decls)
mlir_tablegen(LinalgTransformOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRLinalgTransformOpsIncGen)

add_mlir_doc(LinalgTransformOps LinalgStructuredTransformOps Dialects/ -gen-op-doc)
