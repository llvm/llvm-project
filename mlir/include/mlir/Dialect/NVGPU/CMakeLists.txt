add_mlir_dialect(NVGPU nvgpu)
add_mlir_doc(NVGPU NVGPU Dialects/ -gen-dialect-doc)


set(LLVM_TARGET_DEFINITIONS NVGPU.td)
