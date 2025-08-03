# Exclude tests from libMLIR.so
add_mlir_library(MLIRShardTest
  TestOpLowering.cpp
  TestReshardingPartition.cpp
  TestSimplifications.cpp

  EXCLUDE_FROM_LIBMLIR
  )
mlir_target_link_libraries(MLIRShardTest PUBLIC
  MLIRShardDialect
  MLIRShardTransforms
  MLIRPass
  MLIRRewrite
  MLIRTransformUtils
  )
