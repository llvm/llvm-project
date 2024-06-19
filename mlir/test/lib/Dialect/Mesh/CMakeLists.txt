# Exclude tests from libMLIR.so
add_mlir_library(MLIRMeshTest
  TestOpLowering.cpp
  TestReshardingSpmdization.cpp
  TestSimplifications.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  MLIRMeshDialect
  MLIRMeshTransforms
  MLIRPass
  MLIRRewrite
  MLIRTransformUtils
  )
