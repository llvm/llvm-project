# Exclude tests from libMLIR.so
add_mlir_library(MLIRNVGPUTestPasses
  TestNVGPUTransforms.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRAffineDialect
  MLIRAnalysis
  MLIRFuncDialect
  MLIRGPUDialect
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIRNVGPUDialect
  MLIRNVGPUTransforms
  MLIRPass
  MLIRSCFDialect
  MLIRTransformUtils
  )
  
