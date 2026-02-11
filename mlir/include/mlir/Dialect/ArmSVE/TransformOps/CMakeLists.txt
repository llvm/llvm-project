set(LLVM_TARGET_DEFINITIONS ArmSVEVectorTransformOps.td)
mlir_tablegen(ArmSVEVectorTransformOps.h.inc -gen-op-decls)
mlir_tablegen(ArmSVEVectorTransformOps.cpp.inc -gen-op-defs)
add_mlir_dialect_tablegen_target(MLIRArmSVEVectorTransformOpsIncGen)

add_mlir_doc(ArmSVEVectorTransformOps ArmSVEVectorTransformOps Dialects/ -gen-op-doc)
