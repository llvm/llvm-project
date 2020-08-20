set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Standard)
add_public_tablegen_target(MLIRStandardTransformsIncGen)

add_mlir_doc(Passes -gen-pass-doc StandardPasses ./)
