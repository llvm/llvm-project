set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name ArmSVE)
add_public_tablegen_target(MLIRArmSVEPassIncGen)

add_mlir_doc(Passes ArmSVEPasses ./ -gen-pass-doc)
