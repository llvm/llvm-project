set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Math)
add_public_tablegen_target(MLIRMathTransformsIncGen)

add_mlir_doc(Passes MathPasses ./ -gen-pass-doc)
