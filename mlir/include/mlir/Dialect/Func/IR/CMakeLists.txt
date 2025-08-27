set(LLVM_TARGET_DEFINITIONS FuncOps.td)
mlir_tablegen(FuncOps.h.inc -gen-op-decls)
mlir_tablegen(FuncOps.cpp.inc -gen-op-defs)
mlir_tablegen(FuncOpsDialect.h.inc -gen-dialect-decls)
mlir_tablegen(FuncOpsDialect.cpp.inc -gen-dialect-defs)
add_mlir_dialect_tablegen_target(MLIRFuncOpsIncGen)

add_mlir_doc(FuncOps FuncOps Dialects/ -gen-op-doc)
