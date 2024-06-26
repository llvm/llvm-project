set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name OpenACC)
add_public_tablegen_target(MLIROpenACCPassIncGen)

add_mlir_doc(Passes OpenACCPasses ./ -gen-pass-doc)
