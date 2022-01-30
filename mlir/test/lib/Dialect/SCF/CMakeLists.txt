# Exclude tests from libMLIR.so
add_mlir_library(MLIRSCFTestPasses
  TestLoopParametricTiling.cpp
  TestLoopUnrolling.cpp
  TestSCFUtils.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRSCF
  MLIRSCFTransforms
  MLIRSCFUtils
  MLIRTransformUtils
  )
