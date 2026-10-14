set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name EmitC)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix EmitC)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix EmitC)
add_mlir_dialect_tablegen_target(MLIREmitCTransformsIncGen)

add_mlir_doc(Passes EmitCPasses ./ -gen-pass-doc)
