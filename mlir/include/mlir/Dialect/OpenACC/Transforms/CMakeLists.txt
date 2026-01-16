set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name OpenACC)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix OpenACC)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix OpenACC)
add_mlir_dialect_tablegen_target(MLIROpenACCPassIncGen)

add_mlir_doc(Passes OpenACCPasses ./ -gen-pass-doc)
