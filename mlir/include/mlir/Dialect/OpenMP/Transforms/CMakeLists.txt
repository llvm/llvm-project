set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name OpenMP)
add_mlir_dialect_tablegen_target(MLIROpenMPPassIncGen)

add_mlir_doc(Passes OpenMPPasses ./ -gen-pass-doc)
