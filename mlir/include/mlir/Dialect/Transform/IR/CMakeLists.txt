# Generate the dialect files from the dialect .td.
#
# TODO: Make it possible to use XDialect instead of XOpsDialect in
# add_mlir_dialect.
set(LLVM_TARGET_DEFINITIONS TransformDialect.td)
mlir_tablegen(TransformDialect.h.inc -gen-dialect-decls -dialect=transform)
mlir_tablegen(TransformDialect.cpp.inc -gen-dialect-defs -dialect=transform)
add_public_tablegen_target(MLIRTransformDialectIncGen)
add_dependencies(mlir-headers MLIRTransformDialectIncGen)

set(LLVM_TARGET_DEFINITIONS TransformTypes.td)
mlir_tablegen(TransformTypes.h.inc -gen-typedef-decls)
mlir_tablegen(TransformTypes.cpp.inc -gen-typedef-defs)
add_public_tablegen_target(MLIRTransformTypesIncGen)
add_dependencies(mlir-headers MLIRTransformTypesIncGen)
add_mlir_doc(TransformTypes TransformTypes Dialects/ -gen-typedef-doc)

set(LLVM_TARGET_DEFINITIONS TransformAttrs.td)
mlir_tablegen(TransformDialectEnums.h.inc -gen-enum-decls)
mlir_tablegen(TransformDialectEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRTransformDialectEnumIncGen)
add_dependencies(mlir-headers MLIRTransformDialectEnumIncGen)

set(LLVM_TARGET_DEFINITIONS TransformAttrs.td)
mlir_tablegen(TransformDialectAttrDefs.h.inc -gen-attrdef-decls)
mlir_tablegen(TransformDialectAttrDefs.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRTransformDialectAttrDefsIncGen)
add_dependencies(mlir-headers MLIRTransformDialectAttrDefsIncGen)

add_mlir_dialect(TransformOps transform)
add_mlir_doc(TransformOps TransformOps Dialects/ -gen-op-doc -dialect=transform)

# Contrary to what the name claims, this only produces the _op_ interface.
add_mlir_interface(TransformInterfaces)
add_mlir_doc(TransformInterfaces TransformOpInterfaces Dialects/ -gen-op-interface-docs)

add_mlir_interface(MatchInterfaces)
add_dependencies(MLIRMatchInterfacesIncGen MLIRTransformInterfacesIncGen)
add_mlir_doc(TransformInterfaces MatchOpInterfaces Dialects/ -gen-op-interface-docs)

set(LLVM_TARGET_DEFINITIONS TransformInterfaces.td)
mlir_tablegen(TransformTypeInterfaces.h.inc -gen-type-interface-decls)
mlir_tablegen(TransformTypeInterfaces.cpp.inc -gen-type-interface-defs)
add_public_tablegen_target(MLIRTransformDialectTypeInterfacesIncGen)
add_dependencies(mlir-headers MLIRTransformDialectTypeInterfacesIncGen)
add_mlir_doc(TransformInterfaces TransformTypeInterfaces Dialects/ -gen-type-interface-docs)
