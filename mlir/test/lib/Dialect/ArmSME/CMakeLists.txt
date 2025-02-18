# Exclude tests from libMLIR.so
add_mlir_library(MLIRArmSMETestPasses
  TestLowerToArmSME.cpp

  EXCLUDE_FROM_LIBMLIR
  )
mlir_target_link_libraries(MLIRArmSMETestPasses PUBLIC
  MLIRArithToArmSME
  MLIRArmSMEToLLVM
  MLIRArmSMEToSCF
  MLIRArmSMETransforms
  MLIRArmSVETransforms
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRVectorToArmSME
  MLIRVectorToSCF
  MLIRSCFToControlFlow
  )
