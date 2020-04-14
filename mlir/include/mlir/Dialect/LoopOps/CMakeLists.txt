add_mlir_dialect(LoopOps loop)
add_mlir_doc(LoopOps -gen-dialect-doc LoopDialect Dialects/)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls)
add_public_tablegen_target(MLIRLoopPassIncGen)

add_mlir_doc(Passes -gen-pass-doc LoopPasses ./)
