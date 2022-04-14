# The dialect does not have its own ops, so just generate the dialect files.
set(LLVM_TARGET_DEFINITIONS TransformDialect.td)
mlir_tablegen(TransformDialect.h.inc -gen-dialect-decls -dialect=transform)
mlir_tablegen(TransformDialect.cpp.inc -gen-dialect-defs -dialect=transform)
add_public_tablegen_target(MLIRTransformDialectIncGen)
add_dependencies(mlir-headers MLIRTransformDialectIncGen)

add_mlir_interface(TransformInterfaces)
