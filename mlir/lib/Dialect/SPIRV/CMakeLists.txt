if (MSVC)
  set_source_files_properties(SPIRVDialect.cpp PROPERTIES COMPILE_FLAGS /bigobj)
endif()

set(LLVM_TARGET_DEFINITIONS SPIRVCanonicalization.td)
mlir_tablegen(SPIRVCanonicalization.inc -gen-rewriters)
add_public_tablegen_target(MLIRSPIRVCanonicalizationIncGen)

add_mlir_dialect_library(MLIRSPIRV
  LayoutUtils.cpp
  SPIRVAttributes.cpp
  SPIRVCanonicalization.cpp
  SPIRVDialect.cpp
  SPIRVOps.cpp
  SPIRVLowering.cpp
  SPIRVTypes.cpp
  TargetAndABI.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/SPIRV

  DEPENDS
  MLIRSPIRVAvailabilityIncGen
  MLIRSPIRVCanonicalizationIncGen
  MLIRSPIRVEnumAvailabilityIncGen
  MLIRSPIRVEnumsIncGen
  MLIRSPIRVOpsIncGen
  MLIRSPIRVOpUtilsGen
  MLIRSPIRVTargetAndABIIncGen

  LINK_LIBS PUBLIC
  MLIRControlFlowInterfaces
  MLIRIR
  MLIRParser
  MLIRSideEffectInterfaces
  MLIRSupport
  MLIRTransforms
  )

add_subdirectory(Serialization)
add_subdirectory(Transforms)
