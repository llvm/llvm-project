add_mlir_dialect(NVGPU nvgpu)
add_mlir_doc(NVGPU NVGPU Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS NVGPU.td)
mlir_tablegen(NVGPUEnums.h.inc -gen-enum-decls)
mlir_tablegen(NVGPUEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRNVGPUEnumsIncGen)

set(LLVM_TARGET_DEFINITIONS NVGPU.td)
mlir_tablegen(NVGPUAttrDefs.h.inc -gen-attrdef-decls)
mlir_tablegen(NVGPUAttrDefs.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRNVGPUAttributesIncGen)

set(LLVM_TARGET_DEFINITIONS NVGPU.td)
mlir_tablegen(NVGPUAttrTypes.h.inc -gen-typedef-decls)
mlir_tablegen(NVGPUAttrTypes.cpp.inc -gen-typedef-decls)
add_public_tablegen_target(MLIRNVGPUTypesIncGen)
