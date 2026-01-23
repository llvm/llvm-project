add_mlir_dialect(NVGPU nvgpu)
add_mlir_doc(NVGPUOps NVGPU Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS NVGPUOps.td)
mlir_tablegen(NVGPUOps.h.inc -gen-op-decls)
mlir_tablegen(NVGPUOps.cpp.inc -gen-op-defs)
add_mlir_dialect_tablegen_target(MLIRNVGPUOpsIncGen)

set(LLVM_TARGET_DEFINITIONS NVGPU.td)
mlir_tablegen(NVGPUEnums.h.inc -gen-enum-decls)
mlir_tablegen(NVGPUEnums.cpp.inc -gen-enum-defs)
add_mlir_dialect_tablegen_target(MLIRNVGPUEnumsIncGen)

set(LLVM_TARGET_DEFINITIONS NVGPU.td)
mlir_tablegen(NVGPUAttrDefs.h.inc -gen-attrdef-decls)
mlir_tablegen(NVGPUAttrDefs.cpp.inc -gen-attrdef-defs)
add_mlir_dialect_tablegen_target(MLIRNVGPUAttributesIncGen)

set(LLVM_TARGET_DEFINITIONS NVGPUTypes.td)
mlir_tablegen(NVGPUTypeDefs.h.inc -gen-typedef-decls)
mlir_tablegen(NVGPUTypeDefs.cpp.inc -gen-typedef-defs)
add_mlir_dialect_tablegen_target(MLIRNVGPUTypesIncGen)
