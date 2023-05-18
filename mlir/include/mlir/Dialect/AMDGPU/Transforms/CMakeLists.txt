set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name AMDGPU)
add_public_tablegen_target(MLIRAMDGPUTransformsIncGen)
add_dependencies(mlir-headers MLIRAMDGPUTransformsIncGen)

add_mlir_doc(Passes AMDGPUPasses ./ -gen-pass-doc)
