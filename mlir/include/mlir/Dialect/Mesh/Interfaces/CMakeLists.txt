set(LLVM_TARGET_DEFINITIONS ShardingInterface.td)
mlir_tablegen(ShardingInterface.h.inc -gen-op-interface-decls)
mlir_tablegen(ShardingInterface.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRShardingInterfaceIncGen)
