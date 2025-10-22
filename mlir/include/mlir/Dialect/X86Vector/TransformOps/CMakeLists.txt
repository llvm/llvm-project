set(LLVM_TARGET_DEFINITIONS X86VectorTransformOps.td)
mlir_tablegen(X86VectorTransformOps.h.inc -gen-op-decls)
mlir_tablegen(X86VectorTransformOps.cpp.inc -gen-op-defs)
add_mlir_dialect_tablegen_target(MLIRX86VectorTransformOpsIncGen)
