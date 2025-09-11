set(LLVM_TARGET_DEFINITIONS SPIRVCanonicalization.td)
mlir_tablegen(SPIRVCanonicalization.inc -gen-rewriters)
add_public_tablegen_target(MLIRSPIRVCanonicalizationIncGen)

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
  MLIRSPIRVImageInterfacesIncGen
  MLIRSPIRVOpsIncGen

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
