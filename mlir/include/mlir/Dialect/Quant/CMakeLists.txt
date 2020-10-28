add_mlir_dialect(QuantOps quant)
add_mlir_doc(QuantOps -gen-dialect-doc QuantDialect Dialects/)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Quant)
add_public_tablegen_target(MLIRQuantPassIncGen)

add_mlir_doc(Passes -gen-pass-doc QuantPasses ./)
