add_mlir_dialect(ArmNeon arm_neon)
add_mlir_doc(ArmNeon ArmNeon Dialects/ -gen-dialect-doc -dialect=arm_neon)

set(LLVM_TARGET_DEFINITIONS ArmNeon.td)
mlir_tablegen(ArmNeonConversions.inc -gen-llvmir-conversions)
add_public_tablegen_target(MLIRArmNeonConversionsIncGen)
