set(LLVM_TARGET_DEFINITIONS GPUTransformOps.td)
mlir_tablegen(GPUTransformOps.h.inc -gen-op-decls)
mlir_tablegen(GPUTransformOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRGPUTransformOpsIncGen)

add_mlir_doc(GPUTransformOps GPUTransformOps Dialects/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS GPUDeviceMappingAttr.td)
mlir_tablegen(GPUDeviceMapperEnums.h.inc -gen-enum-decls)
mlir_tablegen(GPUDeviceMapperEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRGPUDeviceMapperEnumsGen)
