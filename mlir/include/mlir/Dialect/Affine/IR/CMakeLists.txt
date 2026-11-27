add_mlir_dialect(AffineOps affine)
add_mlir_doc(AffineOps AffineOps Dialects/ -gen-op-doc -dialect=affine)

add_mlir_interface(AffineMemoryOpInterfaces)
add_dependencies(MLIRAffineOpsIncGen MLIRAffineMemoryOpInterfacesIncGen)
