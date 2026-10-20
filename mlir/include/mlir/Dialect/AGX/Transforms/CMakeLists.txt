set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name AGX)
add_mlir_dialect_tablegen_target(MLIRAGXTransformsIncGen)

add_mlir_doc(Passes AGXPasses ./ -gen-pass-doc)
