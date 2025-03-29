set(LLVM_TARGET_DEFINITIONS GPUTransformOps.td)
mlir_tablegen(GPUTransformOps.h.inc -gen-op-decls)
mlir_tablegen(GPUTransformOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRGPUTransformOpsIncGen)

add_mlir_doc(GPUTransformOps GPUTransformOps Dialects/ -gen-op-doc)
