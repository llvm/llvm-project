set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name TargetLLVMIRTransforms)
add_mlir_dialect_tablegen_target(MLIRTargetLLVMIRTransformsIncGen)

add_mlir_doc(Passes TargetLLVMIRTransforms ./ -gen-pass-doc)
