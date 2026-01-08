set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name XeGPU)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix XeGPU)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix XeGPU)
add_mlir_dialect_tablegen_target(MLIRXeGPUPassIncGen)

add_mlir_doc(Passes XeGPUPasses ./ -gen-pass-doc)
