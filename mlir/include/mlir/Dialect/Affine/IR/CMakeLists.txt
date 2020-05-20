add_mlir_dialect(AffineOps affine)
add_mlir_doc(AffineOps -gen-op-doc AffineOps Dialects/)

set(LLVM_TARGET_DEFINITIONS AffineMemoryOpInterfaces.td)
mlir_tablegen(AffineMemoryOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(AffineMemoryOpInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRAffineMemoryOpInterfacesIncGen)
add_dependencies(mlir-generic-headers MLIRAffineMemoryOpInterfacesIncGen)

add_dependencies(MLIRAffineOpsIncGen MLIRAffineMemoryOpInterfacesIncGen)
