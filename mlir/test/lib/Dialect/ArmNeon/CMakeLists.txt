# Exclude tests from libMLIR.so
add_mlir_library(MLIRArmNeonTestPasses
  TestLowerToArmNeon.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  MLIRArmNeonDialect
  MLIRArmNeonTransforms
  MLIRIR
  MLIRPass
  MLIRTransforms
  )
