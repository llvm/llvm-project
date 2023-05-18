set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Reducer)
add_public_tablegen_target(MLIRReducerIncGen)

add_mlir_doc(Passes ReducerPasses ./ -gen-pass-doc)
