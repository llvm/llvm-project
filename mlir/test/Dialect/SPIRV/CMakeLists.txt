add_llvm_library(MLIRSPIRVTestPasses
  TestAvailability.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/SPIRV
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/IR
  )

target_link_libraries(MLIRSPIRVTestPasses
  MLIRIR
  MLIRPass
  MLIRSPIRV
  MLIRSupport
  )
