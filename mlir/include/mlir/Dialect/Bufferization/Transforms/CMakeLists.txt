set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Bufferization)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix Bufferization)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix Bufferization)
add_mlir_dialect_tablegen_target(MLIRBufferizationPassIncGen)

add_mlir_doc(Passes BufferizationPasses ./ -gen-pass-doc)
