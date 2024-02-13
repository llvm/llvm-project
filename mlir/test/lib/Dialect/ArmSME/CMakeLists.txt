# Exclude tests from libMLIR.so
add_mlir_library(MLIRArmSMETestPasses
  TestLowerToArmSME.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  MLIRAffineToStandard
  MLIRArithToArmSME
  MLIRArmSMEToLLVM
  MLIRArmSMEToSCF
  MLIRFuncDialect
  MLIRFuncToLLVM
  MLIRIR
  MLIRIndexToLLVM
  MLIRLLVMDialect
  MLIRLinalgTransforms
  MLIRMathToLLVM
  MLIRMemRefToLLVM
  MLIRMemRefTransforms
  MLIRPass
  MLIRReconcileUnrealizedCasts
  MLIRSCFToControlFlow
  MLIRTransforms
  MLIRVectorToLLVMPass
  MLIRVectorToSCF
  )
