# Exclude tests from libMLIR.so
add_mlir_library(MLIRSCFTestPasses
  TestLoopParametricTiling.cpp
  TestLoopUnrolling.cpp
  TestParallelLoopUnrolling.cpp
  TestSCFUtils.cpp
  TestSCFWrapInZeroTripCheck.cpp
  TestUpliftWhileToFor.cpp
  TestWhileOpBuilder.cpp

  EXCLUDE_FROM_LIBMLIR
  )
mlir_target_link_libraries(MLIRSCFTestPasses PUBLIC
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRSCFUtils
  MLIRTransformUtils
  )

# Static loop unrolling test pattern conflicts.
set_source_files_properties(
  TestParallelLoopUnrolling.cpp
  PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
