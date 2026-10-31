set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name GPUROCDL)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix GPUROCDL)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix GPUROCDL)
add_mlir_dialect_tablegen_target(MLIRGPUROCDLPassIncGen)

add_mlir_doc(Passes GPUROCDLPasses ./ -gen-pass-doc)
