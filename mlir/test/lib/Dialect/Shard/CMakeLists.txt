# Exclude tests from libMLIR.so
add_mlir_library(MLIRShardTest
  TestOpLowering.cpp
  TestReshardingPartition.cpp

  EXCLUDE_FROM_LIBMLIR

  # TestReshardingPartition.cpp includes SPIRVOps.h.
  HEADER_LIBS
  MLIRSPIRVDialect
  )
mlir_target_link_libraries(MLIRShardTest PUBLIC
  MLIRShardDialect
  MLIRShardTransforms
  MLIRPass
  MLIRRewrite
  MLIRTransformUtils
  )
