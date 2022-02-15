set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Vector)
add_public_tablegen_target(MLIRVectorTransformsIncGen)

add_mlir_doc(Passes VectorPasses ./ -gen-pass-doc)
