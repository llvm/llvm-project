set(LLVM_TARGET_DEFINITIONS SPIRVCanonicalization.td)
mlir_tablegen(SPIRVCanonicalization.inc -gen-rewriters)
add_public_tablegen_target(MLIRSPIRVCanonicalizationIncGen)

set(LLVM_OPTIONAL_SOURCES
  SPIRVOpDefinitions.cpp
  )

set(LLVM_TARGET_DEFINITIONS SPIRVOpDefinitions.cpp)
foreach(index RANGE 7)
  set(SPIRV_SHARDED_SRC SPIRVOpDefinitions.${index}.cpp)
  list(APPEND SPIRV_SHARDED_SRCS ${SPIRV_SHARDED_SRC})
  tablegen(MLIR_SRC_SHARDER ${SPIRV_SHARDED_SRC}
    -op-shard-index=${index})
endforeach()
add_public_tablegen_target(MLIRSPIRVOpsShardGen)

add_mlir_dialect_library(MLIRSPIRVDialect
  ArmGraphOps.cpp
  AtomicOps.cpp
  CastOps.cpp
  ControlFlowOps.cpp
  CooperativeMatrixOps.cpp
  DotProductOps.cpp
  GroupOps.cpp
  ImageOps.cpp
  MemoryOps.cpp
  MeshOps.cpp
  SPIRVAttributes.cpp
  SPIRVCanonicalization.cpp
  SPIRVGLCanonicalization.cpp
  SPIRVDialect.cpp
  SPIRVEnums.cpp
  SPIRVOpAvailability.cpp
  SPIRVOpDefinition.cpp
  SPIRVOps.cpp
  SPIRVParsingUtils.cpp
  SPIRVTosaOps.cpp
  SPIRVTypes.cpp
  TargetAndABI.cpp
  ${SPIRV_SHARDED_SRCS}

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/SPIRV

  DEPENDS
  MLIRGPUDialect
  MLIRSPIRVAttributeIncGen
  MLIRSPIRVAttrUtilsGen
  MLIRSPIRVAvailabilityIncGen
  MLIRSPIRVCanonicalizationIncGen
  MLIRSPIRVEnumAvailabilityIncGen
  MLIRSPIRVEnumsIncGen
  MLIRSPIRVImageInterfacesIncGen
  MLIRSPIRVOpsIncGen
  MLIRSPIRVOpsShardGen

  LINK_LIBS PUBLIC
  MLIRControlFlowInterfaces
  MLIRFunctionInterfaces
  MLIRIR
  MLIRParser
  MLIRSideEffectInterfaces
  MLIRSPIRVImageInterfaces
  MLIRSupport
  MLIRTransforms
  MLIRUBDialect
)
