set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name GPUSPIRV)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix GPUSPIRV)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix GPUSPIRV)
add_mlir_dialect_tablegen_target(MLIRGPUSPIRVPassIncGen)

add_mlir_doc(Passes GPUSPIRVPasses ./ -gen-pass-doc)
