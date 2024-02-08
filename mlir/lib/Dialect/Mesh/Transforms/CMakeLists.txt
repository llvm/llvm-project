add_mlir_dialect_library(MLIRMeshTransforms
  Simplifications.cpp
  ShardingPropagation.cpp
  Spmdization.cpp
  Transforms.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Mesh

  DEPENDS
  MLIRMeshPassIncGen
  MLIRShardingInterface

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRIR
  MLIRMeshDialect
  MLIRPass
  MLIRShardingInterface
  MLIRSupport
  MLIRTensorDialect
  MLIRTosaShardingInterfaceImpl
)
