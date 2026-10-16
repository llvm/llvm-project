add_mlir_library(MLIRShardingInterface
  ShardingInterface.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Shard

  DEPENDS
  MLIRShardingInterfaceIncGen

  LINK_LIBS PUBLIC
  MLIRDialectUtils
  MLIRIR
  MLIRShardDialect
  MLIRTensorDialect
  MLIRSupport
)
