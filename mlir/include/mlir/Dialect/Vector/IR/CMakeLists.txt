add_mlir_dialect(Vector vector)
add_mlir_doc(VectorOps VectorOps Dialects/ -gen-op-doc)

# Add Vector operations
set(LLVM_TARGET_DEFINITIONS VectorOps.td)
mlir_tablegen(VectorOps.h.inc -gen-op-decls)
mlir_tablegen(VectorOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRVectorOpsIncGen)
add_dependencies(mlir-generic-headers MLIRVectorOpsIncGen)

# Add Vector attributes
set(LLVM_TARGET_DEFINITIONS VectorAttributes.td)
mlir_tablegen(VectorEnums.h.inc -gen-enum-decls)
mlir_tablegen(VectorEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(VectorAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(VectorAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRVectorAttributesIncGen)
add_dependencies(mlir-generic-headers MLIRVectorAttributesIncGen)
