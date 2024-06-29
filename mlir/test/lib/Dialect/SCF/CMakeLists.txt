# Exclude tests from libMLIR.so
add_mlir_library(MLIRSCFTestPasses
  TestLoopParametricTiling.cpp
  TestLoopUnrolling.cpp
  TestSCFUtils.cpp
  TestSCFWrapInZeroTripCheck.cpp
  TestUpliftWhileToFor.cpp
  TestWhileOpBuilder.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRSCFUtils
  MLIRTransformUtils
  )
