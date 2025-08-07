# Exclude tests from libMLIR.so
add_mlir_library(MLIRArmNeonTestPasses
  TestLowerToArmNeon.cpp

  EXCLUDE_FROM_LIBMLIR
  )
mlir_target_link_libraries(MLIRArmNeonTestPasses PUBLIC
  MLIRArmNeonDialect
  MLIRArmNeonTransforms
  MLIRIR
  MLIRPass
  MLIRTransforms
  )
