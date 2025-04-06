add_mlir_library(MLIRShardingInterface
  ShardingInterface.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Mesh

  DEPENDS
  MLIRShardingInterfaceIncGen

  LINK_LIBS PUBLIC
  MLIRDialectUtils
  MLIRIR
  MLIRMeshDialect
  MLIRTensorDialect
  MLIRSupport
)
