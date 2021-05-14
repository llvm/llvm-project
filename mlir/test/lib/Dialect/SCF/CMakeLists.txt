# Exclude tests from libMLIR.so
add_mlir_library(MLIRSCFTestPasses
  TestSCFUtils.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRSCF
  MLIRSCFTransforms
  MLIRTransformUtils
  )
