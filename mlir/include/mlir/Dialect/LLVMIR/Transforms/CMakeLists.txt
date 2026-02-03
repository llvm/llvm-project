set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name LLVM)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix LLVM)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix LLVM)
add_mlir_dialect_tablegen_target(MLIRLLVMPassIncGen)

add_mlir_doc(Passes LLVMPasses ./ -gen-pass-doc)
