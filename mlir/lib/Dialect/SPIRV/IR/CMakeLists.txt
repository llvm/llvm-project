set(LLVM_TARGET_DEFINITIONS SPIRVCanonicalization.td)
mlir_tablegen(SPIRVCanonicalization.inc -gen-rewriters)
add_public_tablegen_target(MLIRSPIRVCanonicalizationIncGen)

add_mlir_dialect_library(MLIRSPIRVDialect
  AtomicOps.cpp
  CastOps.cpp
  ControlFlowOps.cpp
  CooperativeMatrixOps.cpp
  GroupOps.cpp
  IntegerDotProductOps.cpp
  MemoryOps.cpp
  SPIRVAttributes.cpp
  SPIRVCanonicalization.cpp
  SPIRVGLCanonicalization.cpp
  SPIRVDialect.cpp
  SPIRVEnums.cpp
  SPIRVOpAvailability.cpp
  SPIRVOpDefinition.cpp
  SPIRVOps.cpp
  SPIRVParsingUtils.cpp
  SPIRVTypes.cpp
  TargetAndABI.cpp

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
  MLIRSPIRVOpsIncGen

  LINK_LIBS PUBLIC
  MLIRControlFlowInterfaces
  MLIRFunctionInterfaces
  MLIRIR
  MLIRParser
  MLIRSideEffectInterfaces
  MLIRSupport
  MLIRTransforms
  MLIRUBDialect
)
