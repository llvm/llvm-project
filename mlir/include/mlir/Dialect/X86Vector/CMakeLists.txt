add_mlir_dialect(X86Vector x86vector)
add_mlir_doc(X86Vector X86Vector Dialects/ -gen-dialect-doc -dialect=x86vector)

set(LLVM_TARGET_DEFINITIONS X86Vector.td)
mlir_tablegen(X86VectorConversions.inc -gen-llvmir-conversions)
add_public_tablegen_target(MLIRX86VectorConversionsIncGen)
