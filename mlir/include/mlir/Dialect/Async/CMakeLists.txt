add_subdirectory(IR)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Async)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix Async)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix Async)
add_mlir_dialect_tablegen_target(MLIRAsyncPassIncGen)

add_mlir_doc(Passes AsyncPasses ./ -gen-pass-doc)
