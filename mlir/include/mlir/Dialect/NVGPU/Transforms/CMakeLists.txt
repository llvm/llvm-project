set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name NVGPU)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix NVGPU)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix NVGPU)
add_public_tablegen_target(MLIRNVGPUPassIncGen)

add_mlir_doc(Passes NVGPUPasses ./ -gen-pass-doc)
