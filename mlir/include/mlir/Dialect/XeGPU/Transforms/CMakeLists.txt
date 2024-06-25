set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name XeGPU)
add_public_tablegen_target(MLIRXeGPUPassIncGen)
add_dependencies(mlir-headers MLIRXeGPUPassIncGen)

add_mlir_doc(Passes XeGPUPasses ./ -gen-pass-doc)
