set(LLVM_TARGET_DEFINITIONS QuantumRewrites.td)
mlir_tablegen(QuantumRewrites.h.inc -gen-rewriters)
add_public_tablegen_target(MLIRQuantumTransformsRewritesPassIncGen)

set(LLVM_TARGET_DEFINITIONS Convert1QToU.td)
mlir_tablegen(Convert1QToU.h.inc -gen-rewriters)
add_public_tablegen_target(MLIRQuantumConvert1QToUPassIncGen)

set(LLVM_TARGET_DEFINITIONS PrepareForZX.td)
mlir_tablegen(PrepareForZX.h.inc -gen-rewriters)
add_public_tablegen_target(MLIRQuantumPrepareForZXPassIncGen)
