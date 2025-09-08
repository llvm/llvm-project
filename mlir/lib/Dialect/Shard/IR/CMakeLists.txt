add_mlir_dialect_library(MLIRShardDialect
  ShardOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Shard

  DEPENDS
  MLIRShardIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRDialectUtils
  MLIRIR
  MLIRSupport
  MLIRViewLikeInterface
  MLIRDestinationStyleOpInterface
)
