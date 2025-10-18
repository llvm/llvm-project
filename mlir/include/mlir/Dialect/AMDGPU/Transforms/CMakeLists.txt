set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name AMDGPU)
add_mlir_dialect_tablegen_target(MLIRAMDGPUTransformsIncGen)

add_mlir_doc(Passes AMDGPUPasses ./ -gen-pass-doc)
