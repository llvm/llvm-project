set(LLVM_TARGET_DEFINITIONS StructuredOpsUtils.td)
mlir_tablegen(DialectUtilsEnums.h.inc -gen-enum-decls)
mlir_tablegen(DialectUtilsEnums.cpp.inc -gen-enum-defs)
add_mlir_dialect_tablegen_target(MLIRDialectUtilsIncGen)
