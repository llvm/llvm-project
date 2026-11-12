set(LLVM_TARGET_DEFINITIONS X86TransformOps.td)
mlir_tablegen(X86TransformOps.h.inc -gen-op-decls)
mlir_tablegen(X86TransformOps.cpp.inc -gen-op-defs)
add_mlir_dialect_tablegen_target(MLIRX86TransformOpsIncGen)
