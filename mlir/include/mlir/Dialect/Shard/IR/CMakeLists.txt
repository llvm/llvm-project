add_mlir_doc(ShardOps ShardOps Dialects/ -gen-op-doc -dialect=shard)
add_mlir_doc(ShardOps ShardAttrs Dialects/ -gen-attrdef-doc -dialect=shard)

set(LLVM_TARGET_DEFINITIONS ShardOps.td)
mlir_tablegen(ShardDialect.cpp.inc -gen-dialect-defs -dialect=shard)
mlir_tablegen(ShardDialect.h.inc -gen-dialect-decls -dialect=shard)

set(LLVM_TARGET_DEFINITIONS ShardBase.td)
mlir_tablegen(ShardAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(ShardAttributes.cpp.inc -gen-attrdef-defs)

set(LLVM_TARGET_DEFINITIONS ShardBase.td)
mlir_tablegen(ShardEnums.h.inc -gen-enum-decls)
mlir_tablegen(ShardEnums.cpp.inc -gen-enum-defs)

set(LLVM_TARGET_DEFINITIONS ShardBase.td)
mlir_tablegen(ShardTypes.h.inc -gen-typedef-decls)
mlir_tablegen(ShardTypes.cpp.inc -gen-typedef-defs)

set(LLVM_TARGET_DEFINITIONS ShardOps.td)
mlir_tablegen(ShardOps.h.inc -gen-op-decls)
mlir_tablegen(ShardOps.cpp.inc -gen-op-defs)

add_mlir_dialect_tablegen_target(MLIRShardIncGen)
