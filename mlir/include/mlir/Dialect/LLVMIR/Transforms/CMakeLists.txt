set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name LLVM)
add_public_tablegen_target(MLIRLLVMPassIncGen)

add_mlir_doc(Passes -gen-pass-doc LLVMPasses ./)
