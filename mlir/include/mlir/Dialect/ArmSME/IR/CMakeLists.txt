add_mlir_dialect(ArmSME arm_sme ArmSME)

# Generate declarations and definitions of ArmSME Ops, the `Core` op partition
# of ArmSMEAllOps.td.
set(LLVM_TARGET_DEFINITIONS ArmSMEAllOps.td)
mlir_tablegen(ArmSMEOps.h.inc -gen-op-decls -op-partition=Core)
mlir_tablegen(ArmSMEOps.cpp.inc -gen-op-defs -op-partition=Core)
set(LLVM_TARGET_DEFINITIONS ArmSMEOps.td)
mlir_tablegen(ArmSMEEnums.h.inc -gen-enum-decls)
mlir_tablegen(ArmSMEEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(ArmSMEAttrDefs.h.inc -gen-attrdef-decls -attrdefs-dialect=arm_sme)
mlir_tablegen(ArmSMEAttrDefs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=arm_sme)
add_mlir_dialect_tablegen_target(MLIRArmSMEOpsIncGen)

# Generate LLVM IR Conversions
set(LLVM_TARGET_DEFINITIONS ArmSMEOps.td)
mlir_tablegen(ArmSMEOpsConversions.inc -gen-llvmir-conversions)
add_mlir_dialect_tablegen_target(MLIRArmSMEConversionsIncGen)

# Generate op interface declarations and definitions
set(LLVM_TARGET_DEFINITIONS ArmSMEOps.td)
mlir_tablegen(ArmSMEOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(ArmSMEOpInterfaces.cpp.inc -gen-op-interface-defs)
add_mlir_dialect_tablegen_target(MLIRArmSMEOpInterfaces)

# Generate declarations and definitions of ArmSME intrinsic Ops, the
# `Intrinsic` op partition of ArmSMEAllOps.td.
set(LLVM_TARGET_DEFINITIONS ArmSMEAllOps.td)
mlir_tablegen(ArmSMEIntrinsicOps.h.inc -gen-op-decls -op-partition=Intrinsic)
mlir_tablegen(ArmSMEIntrinsicOps.cpp.inc -gen-op-defs -op-partition=Intrinsic)
set(LLVM_TARGET_DEFINITIONS ArmSMEIntrinsicOps.td)
mlir_tablegen(ArmSMEIntrinsicConversions.inc -gen-llvmir-conversions)
add_mlir_dialect_tablegen_target(MLIRArmSMEIntrinsicOpsIncGen)

# Generate the docs
add_mlir_doc(ArmSMEOps ArmSMEOps Dialects/ -gen-op-doc -dialect=arm_sme)
add_mlir_doc(ArmSMEIntrinsicOps ArmSMEIntrinsicOps Dialects/ -gen-op-doc -dialect=arm_sme)
