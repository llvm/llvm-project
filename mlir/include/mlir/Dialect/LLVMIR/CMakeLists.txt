add_subdirectory(Transforms)

set(LLVM_TARGET_DEFINITIONS LLVMAttrDefs.td)
mlir_tablegen(LLVMOpsAttrDefs.h.inc -gen-attrdef-decls)
mlir_tablegen(LLVMOpsAttrDefs.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRLLVMAttrsIncGen)

set(LLVM_TARGET_DEFINITIONS LLVMOps.td)
mlir_tablegen(LLVMOps.h.inc -gen-op-decls)
mlir_tablegen(LLVMOps.cpp.inc -gen-op-defs)
mlir_tablegen(LLVMOpsDialect.h.inc -gen-dialect-decls)
mlir_tablegen(LLVMOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(LLVMOpsEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRLLVMOpsIncGen)

add_mlir_doc(LLVMOps -gen-op-doc LLVMOps Dialects/)

add_mlir_interface(LLVMOpsInterfaces)

set(LLVM_TARGET_DEFINITIONS LLVMOps.td)
mlir_tablegen(LLVMConversions.inc -gen-llvmir-conversions)
mlir_tablegen(LLVMConversionEnumsToLLVM.inc -gen-enum-to-llvmir-conversions)
mlir_tablegen(LLVMConversionEnumsFromLLVM.inc -gen-enum-from-llvmir-conversions)
add_public_tablegen_target(MLIRLLVMConversionsIncGen)

add_mlir_dialect(NVVMOps nvvm)
add_mlir_doc(NVVMOps -gen-dialect-doc NVVMDialect Dialects/)
set(LLVM_TARGET_DEFINITIONS NVVMOps.td)
mlir_tablegen(NVVMConversions.inc -gen-llvmir-conversions)
add_public_tablegen_target(MLIRNVVMConversionsIncGen)

add_mlir_dialect(ROCDLOps rocdl)
add_mlir_doc(ROCDLOps -gen-dialect-doc ROCDLDialect Dialects/)
set(LLVM_TARGET_DEFINITIONS ROCDLOps.td)
mlir_tablegen(ROCDLConversions.inc -gen-llvmir-conversions)
add_public_tablegen_target(MLIRROCDLConversionsIncGen)

add_mlir_dialect(LLVMArmSVE llvm_arm_sve LLVMArmSVE)
add_mlir_doc(LLVMArmSVE -gen-dialect-doc LLVMArmSve Dialects/)
set(LLVM_TARGET_DEFINITIONS LLVMArmSVE.td)
mlir_tablegen(LLVMArmSVEConversions.inc -gen-llvmir-conversions)
add_public_tablegen_target(MLIRLLVMArmSVEConversionsIncGen)
