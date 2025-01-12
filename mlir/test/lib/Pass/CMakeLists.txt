get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
# Exclude tests from libMLIR.so
add_mlir_library(MLIRTestPass
  TestDynamicPipeline.cpp
  TestPassManager.cpp
  TestSPIRVCPURunnerPipeline.cpp
  TestVulkanRunnerPipeline.cpp

  EXCLUDE_FROM_LIBMLIR

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Pass

  LINK_LIBS PUBLIC
  ${conversion_libs}
  MLIRIR
  MLIRPass
  MLIRTestDialect
  )

target_include_directories(MLIRTestPass
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../Dialect/Test
  ${CMAKE_CURRENT_BINARY_DIR}/../Dialect/Test
  )
