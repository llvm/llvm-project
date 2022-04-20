set(LLVM_TARGET_DEFINITIONS MLProgramOps.td)
add_mlir_dialect(MLProgramOps ml_program)
add_mlir_doc(MLProgramOps MLProgramOps Dialects/ -gen-dialect-doc)
