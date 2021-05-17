add_mlir_dialect(QuantOps quant)
add_mlir_doc(QuantOps QuantDialect Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Quant)
add_public_tablegen_target(MLIRQuantPassIncGen)

add_mlir_doc(Passes QuantPasses ./ -gen-pass-doc)
