add_mlir_dialect(ArmSVE arm_sve ArmSVE)
add_mlir_doc(ArmSVE ArmSVE Dialects/ -gen-dialect-doc -dialect=arm_sve)

set(LLVM_TARGET_DEFINITIONS ArmSVE.td)
mlir_tablegen(ArmSVEConversions.inc -gen-llvmir-conversions)
add_mlir_dialect_tablegen_target(MLIRArmSVEConversionsIncGen)
