add_subdirectory(IR)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Affine)
add_public_tablegen_target(MLIRAffinePassIncGen)

add_mlir_doc(Passes -gen-pass-doc AffinePasses ./)
