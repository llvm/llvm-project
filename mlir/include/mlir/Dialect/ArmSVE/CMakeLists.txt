add_mlir_dialect(ArmSVE arm_sve ArmSVE)
add_mlir_doc(ArmSVE ArmSVE Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS ArmSVE.td)
mlir_tablegen(ArmSVEConversions.inc -gen-llvmir-conversions)
add_public_tablegen_target(MLIRArmSVEConversionsIncGen)
