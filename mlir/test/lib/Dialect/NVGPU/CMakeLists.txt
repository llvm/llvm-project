# Exclude tests from libMLIR.so
add_mlir_library(MLIRNVGPUTestPasses
  TestNVGPUTransforms.cpp

  EXCLUDE_FROM_LIBMLIR
  )
mlir_target_link_libraries(MLIRNVGPUTestPasses PUBLIC
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
  
