set(LLVM_TARGET_DEFINITIONS TosaOps.td)
mlir_tablegen(TosaDialect.h.inc -gen-dialect-decls)
mlir_tablegen(TosaOps.h.inc -gen-op-decls)
mlir_tablegen(TosaOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRTosaOpsIncGen)

set(LLVM_TARGET_DEFINITIONS TosaOps.td)
mlir_tablegen(TosaStructs.h.inc -gen-struct-attr-decls)
mlir_tablegen(TosaStructs.cpp.inc -gen-struct-attr-defs)
add_public_tablegen_target(MLIRTosaStructsIncGen)


set(LLVM_TARGET_DEFINITIONS TosaInterfaces.td)
mlir_tablegen(TosaInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(TosaInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRTosaInterfaceIncGen)

add_mlir_doc(TosaOps -gen-op-doc TosaOps Dialects/)
