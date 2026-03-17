set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Quant)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix Quant)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix Quant)
add_mlir_dialect_tablegen_target(MLIRQuantTransformsIncGen)

add_mlir_doc(Passes QuantPasses ./ -gen-pass-doc)
