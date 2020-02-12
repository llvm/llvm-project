set(LLVM_TARGET_DEFINITIONS SPIRVCanonicalization.td)
mlir_tablegen(SPIRVCanonicalization.inc -gen-rewriters)
add_public_tablegen_target(MLIRSPIRVCanonicalizationIncGen)

add_llvm_library(MLIRSPIRV
  LayoutUtils.cpp
  SPIRVDialect.cpp
  SPIRVOps.cpp
  SPIRVLowering.cpp
  SPIRVTypes.cpp
  TargetAndABI.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/SPIRV
  )

add_dependencies(MLIRSPIRV
  MLIRSPIRVAvailabilityIncGen
  MLIRSPIRVCanonicalizationIncGen
  MLIRSPIRVEnumAvailabilityIncGen
  MLIRSPIRVEnumsIncGen
  MLIRSPIRVOpsIncGen
  MLIRSPIRVOpUtilsGen
  MLIRSPIRVTargetAndABIIncGen)

target_link_libraries(MLIRSPIRV
  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRTransforms)

add_subdirectory(Serialization)
add_subdirectory(Transforms)
