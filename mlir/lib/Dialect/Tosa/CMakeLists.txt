add_mlir_dialect_library(MLIRTosaDialect
  IR/TosaOps.cpp
  IR/TosaCanonicalizations.cpp
  IR/TargetEnv.cpp
  Utils/ConversionUtils.cpp
  Utils/QuantUtils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Tosa

  DEPENDS
  MLIRTosaAttributesIncGen
  MLIRTosaDialectBytecodeIncGen
  MLIRTosaOpsIncGen
  MLIRTosaInterfacesIncGen
  MLIRTosaEnumsIncGen
  MLIRShardingInterfaceIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRDialect
  MLIRCallInterfaces
  MLIRControlFlowInterfaces
  MLIRQuantUtils
  MLIRSideEffectInterfaces
  MLIRTensorDialect
  MLIRVectorInterfaces
  MLIRViewLikeInterface
  )

add_subdirectory(Extensions)
add_subdirectory(Transforms)
