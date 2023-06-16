add_mlir_dialect(ArmSME arm_sme ArmSME)
add_mlir_doc(ArmSME ArmSME Dialects/ -gen-dialect-doc -dialect=arm_sme)

set(LLVM_TARGET_DEFINITIONS ArmSME.td)
mlir_tablegen(ArmSMEConversions.inc -gen-llvmir-conversions)
add_public_tablegen_target(MLIRArmSMEConversionsIncGen)
