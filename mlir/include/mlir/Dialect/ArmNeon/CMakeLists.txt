add_mlir_dialect(ArmNeon arm_neon)
add_mlir_doc(ArmNeon -gen-dialect-doc ArmNeon Dialects/)

set(LLVM_TARGET_DEFINITIONS ArmNeon.td)
mlir_tablegen(ArmNeonConversions.inc -gen-llvmir-conversions)
add_public_tablegen_target(MLIRArmNeonConversionsIncGen)
