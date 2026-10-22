set(LLVM_TARGET_DEFINITIONS PolygeistOps.td)
add_mlir_dialect(PolygeistOps polygeist)
add_mlir_doc(PolygeistOps PolygeistOps Dialects/ -gen-dialect-doc)
