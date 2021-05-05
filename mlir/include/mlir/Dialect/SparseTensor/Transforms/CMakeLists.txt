set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name SparseTensor)
add_public_tablegen_target(MLIRSparseTensorPassIncGen)

add_mlir_doc(Passes SparseTensorPasses ./ -gen-pass-doc)
