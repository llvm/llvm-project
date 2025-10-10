set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Bufferization)
add_mlir_dialect_tablegen_target(MLIRBufferizationPassIncGen)

add_mlir_doc(Passes BufferizationPasses ./ -gen-pass-doc)
