add_mlir_dialect(ArmSME arm_sme ArmSME)
add_mlir_doc(ArmSME ArmSME Dialects/ -gen-dialect-doc -dialect=arm_sme)

set(LLVM_TARGET_DEFINITIONS ArmSME.td)
mlir_tablegen(ArmSMEConversions.inc -gen-llvmir-conversions)
add_public_tablegen_target(MLIRArmSMEConversionsIncGen)

mlir_tablegen(ArmSMEEnums.h.inc -gen-enum-decls)
mlir_tablegen(ArmSMEEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(ArmSMEAttrDefs.h.inc -gen-attrdef-decls -attrdefs-dialect=arm_sme)
mlir_tablegen(ArmSMEAttrDefs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=arm_sme)
add_public_tablegen_target(MLIRArmSMEAttrDefsIncGen)
