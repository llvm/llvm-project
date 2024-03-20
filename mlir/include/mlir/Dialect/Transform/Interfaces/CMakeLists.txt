# Contrary to what the name claims, this only produces the _op_ interface. The
# type interface is produced manually below.
add_mlir_interface(TransformInterfaces)
add_mlir_doc(TransformInterfaces TransformOpInterfaces Dialects/ -gen-op-interface-docs)

set(LLVM_TARGET_DEFINITIONS TransformInterfaces.td)
mlir_tablegen(TransformTypeInterfaces.h.inc -gen-type-interface-decls)
mlir_tablegen(TransformTypeInterfaces.cpp.inc -gen-type-interface-defs)
add_public_tablegen_target(MLIRTransformDialectTypeInterfacesIncGen)
add_dependencies(mlir-headers MLIRTransformDialectTypeInterfacesIncGen)
add_mlir_doc(TransformInterfaces TransformTypeInterfaces Dialects/ -gen-type-interface-docs)
