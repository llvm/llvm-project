add_mlir_dialect(MeshOps mesh)
add_mlir_doc(MeshOps MeshOps Dialects/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS MeshBase.td)
mlir_tablegen(MeshOpsAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(MeshOpsAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRMeshOpsAttrIncGen)
add_mlir_doc(MeshOps MeshAttributes Dialects/ -gen-attrdef-doc)

set(LLVM_TARGET_DEFINITIONS MeshBase.td)
mlir_tablegen(MeshOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(MeshOpsEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRMeshOpsEnumsIncGen)
