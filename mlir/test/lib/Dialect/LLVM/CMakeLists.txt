# Exclude tests from libMLIR.so
add_mlir_library(MLIRLLVMTestPasses
  TestLowerToLLVM.cpp
  TestPatterns.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  MLIRAffineToStandard
  MLIRFuncDialect
  MLIRFuncToLLVM
  MLIRIndexToLLVM
  MLIRIR
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
