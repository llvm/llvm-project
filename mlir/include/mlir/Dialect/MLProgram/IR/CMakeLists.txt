set(LLVM_TARGET_DEFINITIONS MLProgramOps.td)
add_mlir_dialect(MLProgramOps ml_program)
add_mlir_doc(MLProgramOps MLProgramOps Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS MLProgramAttributes.td)
mlir_tablegen(MLProgramAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(MLProgramAttributes.cpp.inc -gen-attrdef-defs)
add_mlir_dialect_tablegen_target(MLIRMLProgramAttributesIncGen)

set(LLVM_TARGET_DEFINITIONS MLProgramTypes.td)
mlir_tablegen(MLProgramTypes.h.inc -gen-typedef-decls)
mlir_tablegen(MLProgramTypes.cpp.inc -gen-typedef-defs)
add_mlir_dialect_tablegen_target(MLIRMLProgramTypesIncGen)
