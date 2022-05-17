add_mlir_dialect(AMX amx)
add_mlir_doc(AMX AMX Dialects/ -gen-dialect-doc -dialect=amx)

set(LLVM_TARGET_DEFINITIONS AMX.td)
mlir_tablegen(AMXConversions.inc -gen-llvmir-conversions)
add_public_tablegen_target(MLIRAMXConversionsIncGen)
