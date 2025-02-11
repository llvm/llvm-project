add_mlir_dialect(TensorOps tensor)
add_mlir_doc(TensorOps TensorOps Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS TensorInterfaces.td)
mlir_tablegen(TensorInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(TensorInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRTensorInterfacesIncGen)
add_dependencies(mlir-headers MLIRTensorInterfacesIncGen)
