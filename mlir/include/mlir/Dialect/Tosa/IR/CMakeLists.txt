add_mlir_dialect(TosaOps tosa)
add_mlir_doc(TosaOps TosaOps Dialects/ -gen-op-doc)
add_mlir_interface(TosaInterfaces)

set(LLVM_TARGET_DEFINITIONS TosaOps.td)
mlir_tablegen(TosaStructs.h.inc -gen-struct-attr-decls)
mlir_tablegen(TosaStructs.cpp.inc -gen-struct-attr-defs)
add_public_tablegen_target(MLIRTosaStructsIncGen)

