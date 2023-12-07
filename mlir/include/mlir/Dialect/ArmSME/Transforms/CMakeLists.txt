set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name ArmSME)
add_public_tablegen_target(MLIRArmSMETransformsIncGen)

add_mlir_doc(Passes ArmSMEPasses ./ -gen-pass-doc)
