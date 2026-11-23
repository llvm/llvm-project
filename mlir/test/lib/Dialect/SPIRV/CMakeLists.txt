# Exclude tests from libMLIR.so
add_mlir_library(MLIRSPIRVTestPasses
  TestAvailability.cpp
  TestEntryPointAbi.cpp
  TestModuleCombiner.cpp

  EXCLUDE_FROM_LIBMLIR

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/SPIRV
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/IR
  )
mlir_target_link_libraries(MLIRSPIRVTestPasses PUBLIC
  MLIRGPUDialect
  MLIRIR
  MLIRPass
  MLIRSPIRVDialect
  MLIRSPIRVConversion
  MLIRSPIRVModuleCombiner
  MLIRSupport
  )
