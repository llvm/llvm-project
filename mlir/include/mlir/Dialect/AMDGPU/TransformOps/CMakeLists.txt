set(LLVM_TARGET_DEFINITIONS AMDGPUTransformOps.td)
mlir_tablegen(AMDGPUTransformOps.h.inc -gen-op-decls)
mlir_tablegen(AMDGPUTransformOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRAMDGPUTransformOpsIncGen) 
