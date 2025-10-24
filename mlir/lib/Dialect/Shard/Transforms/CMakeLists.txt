add_mlir_dialect_library(MLIRShardTransforms
  Simplifications.cpp
  ShardingPropagation.cpp
  Partition.cpp
  Transforms.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Shard

  DEPENDS
  MLIRShardPassIncGen
  MLIRShardingInterface

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRAffineUtils
  MLIRArithDialect
  MLIRArithUtils
  MLIRControlFlowDialect
  MLIRControlFlowInterfaces
  MLIRFuncDialect
  MLIRFunctionInterfaces
  MLIRIR
  MLIRShardDialect
  MLIRPass
  MLIRSupport
  MLIRTensorDialect
  MLIRTosaShardingInterfaceImpl
)
