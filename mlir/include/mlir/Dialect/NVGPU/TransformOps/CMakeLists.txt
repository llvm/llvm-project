set(LLVM_TARGET_DEFINITIONS NVGPUTransformOps.td)
mlir_tablegen(NVGPUTransformOps.h.inc -gen-op-decls)
mlir_tablegen(NVGPUTransformOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRNVGPUTransformOpsIncGen)
