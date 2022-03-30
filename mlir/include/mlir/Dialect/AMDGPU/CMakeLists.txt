add_mlir_dialect(AMDGPU amdgpu)
add_mlir_doc(AMDGPU AMDGPU Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS AMDGPU.td)
