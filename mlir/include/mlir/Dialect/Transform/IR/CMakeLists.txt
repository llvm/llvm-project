# Generate the dialect files from the dialect .td.
#
# TODO: Make it possible to use XDialect instead of XOpsDialect in
# add_mlir_dialect.
set(LLVM_TARGET_DEFINITIONS TransformDialect.td)
mlir_tablegen(TransformDialect.h.inc -gen-dialect-decls -dialect=transform)
mlir_tablegen(TransformDialect.cpp.inc -gen-dialect-defs -dialect=transform)
add_public_tablegen_target(MLIRTransformDialectIncGen)
add_dependencies(mlir-headers MLIRTransformDialectIncGen)

set(LLVM_TARGET_DEFINITIONS TransformAttrs.td)
mlir_tablegen(TransformDialectEnums.h.inc -gen-enum-decls)
mlir_tablegen(TransformDialectEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRTransformDialectEnumIncGen)
add_dependencies(mlir-headers MLIRTransformDialectEnumIncGen)

add_mlir_dialect(TransformOps transform)
add_mlir_doc(TransformOps TransformOps Dialects/ -gen-dialect-doc -dialect=transform)

add_mlir_interface(TransformInterfaces)
add_mlir_doc(TransformInterfaces TransformOpInterfaces Dialects/ -gen-op-interface-docs)
