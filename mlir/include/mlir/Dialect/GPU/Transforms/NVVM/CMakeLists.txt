set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name GPUNVVM)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix GPUNVVM)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix GPUNVVM)
add_mlir_dialect_tablegen_target(MLIRGPUNVVMPassIncGen)

add_mlir_doc(Passes GPUNVVMPasses ./ -gen-pass-doc)
