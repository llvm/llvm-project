# Exclude tests from libMLIR.so
add_mlir_library(MLIRArmSMETestPasses
  TestLowerToArmSME.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
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
