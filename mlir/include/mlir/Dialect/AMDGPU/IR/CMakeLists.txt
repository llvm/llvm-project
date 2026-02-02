add_mlir_dialect(AMDGPU amdgpu)
add_mlir_doc(AMDGPU AMDGPU Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS AMDGPUEnums.td)
mlir_tablegen(AMDGPUEnums.h.inc -gen-enum-decls)
mlir_tablegen(AMDGPUEnums.cpp.inc -gen-enum-defs)
add_mlir_dialect_tablegen_target(MLIRAMDGPUEnumsGen)

set(LLVM_TARGET_DEFINITIONS AMDGPUAttrs.td)
mlir_tablegen(AMDGPUAttrs.h.inc -gen-attrdef-decls -attrdefs-dialect=amdgpu)
mlir_tablegen(AMDGPUAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=amdgpu)
add_mlir_dialect_tablegen_target(MLIRAMDGPUAttributesIncGen)
