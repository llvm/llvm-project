set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Arith)
add_public_tablegen_target(MLIRArithTransformsIncGen)

add_mlir_doc(Passes ArithPasses ./ -gen-pass-doc)
