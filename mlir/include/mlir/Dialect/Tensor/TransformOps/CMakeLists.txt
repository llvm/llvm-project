set(LLVM_TARGET_DEFINITIONS TensorTransformOps.td)
mlir_tablegen(TensorTransformOps.h.inc -gen-op-decls)
mlir_tablegen(TensorTransformOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRTensorTransformOpsIncGen)

add_mlir_doc(TensorTransformOps TensorTransformOps Dialects/ -gen-op-doc)
