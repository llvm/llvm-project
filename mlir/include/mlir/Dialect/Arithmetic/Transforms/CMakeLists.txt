set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Arithmetic)
add_public_tablegen_target(MLIRArithmeticTransformsIncGen)

add_mlir_doc(Passes ArithmeticPasses ./ -gen-pass-doc)
