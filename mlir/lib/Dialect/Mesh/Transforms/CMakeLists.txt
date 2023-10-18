add_mlir_dialect_library(MLIRMeshTransforms
  ShardingPropagation.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Mesh

  DEPENDS
  MLIRMeshPassIncGen
  MLIRShardingInterface

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRMeshDialect
  MLIRShardingInterface
  MLIRSupport
  MLIRTosaShardingInterfaceImpl
)
