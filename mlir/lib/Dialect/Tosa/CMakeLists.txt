add_mlir_dialect_library(MLIRTosaDialect
  IR/TosaOps.cpp
  IR/TosaCanonicalizations.cpp
  Utils/ConversionUtils.cpp
  Utils/QuantUtils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Tosa

  DEPENDS
  MLIRTosaAttributesIncGen
  MLIRTosaDialectBytecodeIncGen
  MLIRTosaOpsIncGen
  MLIRTosaInterfacesIncGen
  MLIRShardingInterfaceIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRDialect
  MLIRCallInterfaces
  MLIRControlFlowInterfaces
  MLIRQuantDialect
  MLIRQuantUtils
  MLIRSideEffectInterfaces
  MLIRTensorDialect
  MLIRVectorInterfaces
  MLIRViewLikeInterface
  )

add_mlir_dialect_library(MLIRTosaShardingInterfaceImpl
  IR/ShardingInterfaceImpl.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Tosa

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRMeshDialect
  MLIRShardingInterface
  MLIRSupport
  MLIRTosaDialect
  )

add_subdirectory(Transforms)
