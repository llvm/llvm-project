set(LLVM_TARGET_DEFINITIONS XeGPUTransformOps.td)
mlir_tablegen(XeGPUTransformOps.h.inc -gen-op-decls)
mlir_tablegen(XeGPUTransformOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRXeGPUTransformOpsIncGen)

add_mlir_doc(XeGPUTransformOps XeGPUTransformOps Dialects/ -gen-op-doc)
