set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name SPIRV)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix SPIRV)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix SPIRV)
add_mlir_dialect_tablegen_target(MLIRSPIRVPassIncGen)

add_mlir_doc(Passes SPIRVPasses ./ -gen-pass-doc)
