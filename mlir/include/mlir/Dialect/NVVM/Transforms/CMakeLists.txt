set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name NVVM)
add_mlir_dialect_tablegen_target(MLIRNVVMPassIncGen)

add_mlir_doc(Passes NVVMPasses ./ -gen-pass-doc)
