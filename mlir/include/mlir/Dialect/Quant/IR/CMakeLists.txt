add_mlir_dialect(QuantOps quant)
add_mlir_doc(QuantOps QuantDialect Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS QuantDialectBytecode.td)
mlir_tablegen(QuantDialectBytecode.cpp.inc -gen-bytecode -bytecode-dialect="Quant")
add_mlir_dialect_tablegen_target(MLIRQuantDialectBytecodeIncGen)
