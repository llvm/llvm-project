set(LLVM_TARGET_DEFINITIONS FuncTransformOps.td)
mlir_tablegen(FuncTransformOps.h.inc -gen-op-decls)
mlir_tablegen(FuncTransformOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRFuncTransformOpsIncGen)

add_mlir_doc(FuncTransformOps FuncTransformOps Dialects/ -gen-op-doc)
