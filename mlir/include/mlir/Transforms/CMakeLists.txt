
set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Transforms)
add_public_tablegen_target(MLIRTransformsPassIncGen)

add_mlir_doc(Passes -gen-pass-doc GeneralPasses ./)
