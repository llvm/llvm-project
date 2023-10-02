add_mlir_dialect(ArmSME arm_sme ArmSME)
add_mlir_doc(ArmSME ArmSME Dialects/ -gen-dialect-doc -dialect=arm_sme)

# Generate declarations and definitions of ArmSME Ops
set(LLVM_TARGET_DEFINITIONS ArmSMEOps.td)
mlir_tablegen(ArmSMEOps.h.inc -gen-op-decls)
mlir_tablegen(ArmSMEOps.cpp.inc -gen-op-defs)
mlir_tablegen(ArmSMEEnums.h.inc -gen-enum-decls)
mlir_tablegen(ArmSMEEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(ArmSMEAttrDefs.h.inc -gen-attrdef-decls -attrdefs-dialect=arm_sme)
mlir_tablegen(ArmSMEAttrDefs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=arm_sme)
add_public_tablegen_target(MLIRArmSMEOpsIncGen)

# Generate LLVM IR Conversions
set(LLVM_TARGET_DEFINITIONS ArmSMEOps.td)
mlir_tablegen(ArmSMEOpsConversions.inc -gen-llvmir-conversions)
add_public_tablegen_target(MLIRArmSMEConversionsIncGen)

# Generate declarations and definitions of ArmSME intrinsic Ops
set(LLVM_TARGET_DEFINITIONS ArmSMEIntrinsicOps.td)
mlir_tablegen(ArmSMEIntrinsicOps.h.inc -gen-op-decls)
mlir_tablegen(ArmSMEIntrinsicOps.cpp.inc -gen-op-defs)
mlir_tablegen(ArmSMEIntrinsicConversions.inc -gen-llvmir-conversions)
add_public_tablegen_target(MLIRArmSMEIntrinsicOpsIncGen)
