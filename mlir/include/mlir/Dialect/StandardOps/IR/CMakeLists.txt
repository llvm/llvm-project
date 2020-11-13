set(LLVM_TARGET_DEFINITIONS Ops.td)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)
mlir_tablegen(OpsDialect.h.inc -gen-dialect-decls)
mlir_tablegen(OpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(OpsEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRStandardOpsIncGen)

add_mlir_doc(Ops -gen-op-doc StandardOps Dialects/)

if (MLIR_BINDINGS_PYTHON_ENABLED)
  add_mlir_dialect_python_bindings(Ops.td std StandardOps)
endif()
