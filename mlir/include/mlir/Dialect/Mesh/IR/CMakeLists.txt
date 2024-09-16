add_mlir_doc(MeshOps MeshOps Dialects/ -gen-op-doc -dialect=mesh)
add_mlir_doc(MeshOps MeshAttrs Dialects/ -gen-attrdef-doc -dialect=mesh)

set(LLVM_TARGET_DEFINITIONS MeshOps.td)
mlir_tablegen(MeshDialect.cpp.inc -gen-dialect-defs -dialect=mesh)
mlir_tablegen(MeshDialect.h.inc -gen-dialect-decls -dialect=mesh)

set(LLVM_TARGET_DEFINITIONS MeshBase.td)
mlir_tablegen(MeshAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(MeshAttributes.cpp.inc -gen-attrdef-defs)

set(LLVM_TARGET_DEFINITIONS MeshBase.td)
mlir_tablegen(MeshEnums.h.inc -gen-enum-decls)
mlir_tablegen(MeshEnums.cpp.inc -gen-enum-defs)

set(LLVM_TARGET_DEFINITIONS MeshBase.td)
mlir_tablegen(MeshTypes.h.inc -gen-typedef-decls)
mlir_tablegen(MeshTypes.cpp.inc -gen-typedef-defs)

set(LLVM_TARGET_DEFINITIONS MeshOps.td)
mlir_tablegen(MeshOps.h.inc -gen-op-decls)
mlir_tablegen(MeshOps.cpp.inc -gen-op-defs)

add_public_tablegen_target(MLIRMeshIncGen)
add_dependencies(mlir-headers MLIRMeshIncGen)
