add_mlir_dialect(FxpMathOps fxpmath)
add_mlir_doc(FxpMathOps -gen-dialect-doc FxpMathDialect Dialects/)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls)
add_public_tablegen_target(MLIRFxpMathPassIncGen)
