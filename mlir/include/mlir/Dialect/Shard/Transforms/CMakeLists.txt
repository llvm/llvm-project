set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Shard)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix Shard)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix Shard)
add_mlir_dialect_tablegen_target(MLIRShardPassIncGen)

add_mlir_doc(Passes ShardPasses ./ -gen-pass-doc)
