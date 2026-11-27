set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name AMDGPU)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix AMDGPU)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix AMDGPU)
add_mlir_dialect_tablegen_target(MLIRAMDGPUTransformsIncGen)

add_mlir_doc(Passes AMDGPUPasses ./ -gen-pass-doc)
