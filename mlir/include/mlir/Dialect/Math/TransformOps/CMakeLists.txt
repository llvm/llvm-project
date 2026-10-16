set(LLVM_TARGET_DEFINITIONS MathTransformOps.td)
mlir_tablegen(MathTransformOps.h.inc -gen-op-decls)
mlir_tablegen(MathTransformOps.cpp.inc -gen-op-defs)
add_mlir_dialect_tablegen_target(MLIRMathTransformOpsIncGen)

add_mlir_doc(MathTransformOps MathTransformOps Dialects/ -gen-op-doc -dialect=transform)
