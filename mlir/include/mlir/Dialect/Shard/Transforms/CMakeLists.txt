set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Shard)
add_public_tablegen_target(MLIRShardPassIncGen)
add_dependencies(mlir-headers MLIRShardPassIncGen)

add_mlir_doc(Passes ShardPasses ./ -gen-pass-doc)
