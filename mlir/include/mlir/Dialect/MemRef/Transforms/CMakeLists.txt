set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name MemRef)
add_mlir_dialect_tablegen_target(MLIRMemRefPassIncGen)

add_mlir_doc(Passes MemRefPasses ./ -gen-pass-doc)
