add_mlir_dialect(AVX512 avx512)
add_mlir_doc(AVX512 -gen-dialect-doc AVX512 Dialects/)

set(LLVM_TARGET_DEFINITIONS AVX512.td)
mlir_tablegen(AVX512Conversions.inc -gen-llvmir-conversions)
add_public_tablegen_target(MLIRAVX512ConversionsIncGen)
