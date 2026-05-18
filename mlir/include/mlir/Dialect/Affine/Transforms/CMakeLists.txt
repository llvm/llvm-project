set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Affine)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix Affine)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix Affine)
add_mlir_dialect_tablegen_target(MLIRAffinePassIncGen)

add_mlir_doc(Passes AffinePasses ./ -gen-pass-doc)
