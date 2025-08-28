set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Quant)
add_mlir_dialect_tablegen_target(MLIRQuantTransformsIncGen)

add_mlir_doc(Passes QuantPasses ./ -gen-pass-doc)
