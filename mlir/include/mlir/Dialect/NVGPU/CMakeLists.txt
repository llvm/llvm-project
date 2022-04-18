add_mlir_dialect(NVGPU nvgpu)
add_mlir_doc(NVGPU -gen-dialect-doc NVGPU Dialects/)

set(LLVM_TARGET_DEFINITIONS NVGPU.td)
