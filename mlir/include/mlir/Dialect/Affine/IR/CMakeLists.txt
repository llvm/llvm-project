add_mlir_dialect(AffineOps affine)
add_mlir_doc(AffineOps -gen-op-doc AffineOps Dialects/)

add_mlir_interface(AffineMemoryOpInterfaces)
add_dependencies(MLIRAffineOpsIncGen MLIRAffineMemoryOpInterfacesIncGen)
