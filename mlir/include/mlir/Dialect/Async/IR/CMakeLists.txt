set(LLVM_TARGET_DEFINITIONS AsyncOps.td)
add_mlir_dialect(AsyncOps async)
add_mlir_doc(AsyncOps AsyncDialect Dialects/ -gen-dialect-doc)
