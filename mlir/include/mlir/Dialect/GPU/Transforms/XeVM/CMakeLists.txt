set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name GPUXeVM)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix GPUXeVM)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix GPUXeVM)
add_mlir_dialect_tablegen_target(MLIRGPUXeVMPassIncGen)

add_mlir_doc(Passes GPUXeVMPasses ./ -gen-pass-doc)
