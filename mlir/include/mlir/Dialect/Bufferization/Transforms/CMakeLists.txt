set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Bufferization)
add_public_tablegen_target(MLIRBufferizationPassIncGen)
add_dependencies(mlir-headers MLIRBufferizationPassIncGen)

add_mlir_doc(Passes BufferizationPasses ./ -gen-pass-doc)
