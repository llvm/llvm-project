set(LLVM_TARGET_DEFINITIONS OpenACCOps.td)
mlir_tablegen(OpenACCOpsDialect.h.inc -gen-dialect-decls -dialect=acc)
mlir_tablegen(OpenACCOps.h.inc -gen-op-decls)
mlir_tablegen(OpenACCOps.cpp.inc -gen-op-defs)
mlir_tablegen(OpenACCOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(OpenACCOpsEnums.cpp.inc -gen-enum-defs)
add_mlir_doc(OpenACCOps -gen-dialect-doc OpenACCDialect Dialects/)
add_public_tablegen_target(MLIROpenACCOpsIncGen)

