add_mlir_dialect(VectorOps vector)
add_mlir_doc(VectorOps VectorOps Dialects/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS VectorOps.td)
mlir_tablegen(VectorOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(VectorOpsEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRVectorOpsEnumsIncGen)
add_dependencies(mlir-headers MLIRVectorOpsEnumsIncGen)
