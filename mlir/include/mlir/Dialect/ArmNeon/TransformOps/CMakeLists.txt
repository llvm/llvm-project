set(LLVM_TARGET_DEFINITIONS ArmNeonVectorTransformOps.td)
mlir_tablegen(ArmNeonVectorTransformOps.h.inc -gen-op-decls)
mlir_tablegen(ArmNeonVectorTransformOps.cpp.inc -gen-op-defs)
add_mlir_dialect_tablegen_target(MLIRArmNeonVectorTransformOpsIncGen)

add_mlir_doc(ArmNeonVectorTransformOps ArmNeonVectorTransformOps Dialects/ -gen-op-doc)
