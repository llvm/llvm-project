set(LLVM_TARGET_DEFINITIONS AMXTransformOps.td)
mlir_tablegen(AMXTransformOps.h.inc -gen-op-decls)
mlir_tablegen(AMXTransformOps.cpp.inc -gen-op-defs)
add_mlir_dialect_tablegen_target(MLIRAMXTransformOpsIncGen)
