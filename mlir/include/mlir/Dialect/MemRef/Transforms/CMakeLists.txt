set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name MemRef)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix MemRef)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix MemRef)
add_mlir_dialect_tablegen_target(MLIRMemRefPassIncGen)

add_mlir_doc(Passes MemRefPasses ./ -gen-pass-doc)
