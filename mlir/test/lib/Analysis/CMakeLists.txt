# Exclude tests from libMLIR.so
add_mlir_library(MLIRTestAnalysis
  TestAliasAnalysis.cpp

  EXCLUDE_FROM_LIBMLIR

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Transforms

  DEPENDS
  MLIRStandardOpsIncGen

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRPass
  MLIRTestDialect
  )

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../Dialect/Test)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../Dialect/Test)
