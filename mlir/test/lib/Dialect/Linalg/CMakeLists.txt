# Exclude tests from libMLIR.so
add_mlir_library(MLIRLinalgTestPasses
  TestLinalgCodegenStrategy.cpp
  TestLinalgElementwiseFusion.cpp
  TestLinalgFusionTransforms.cpp
  TestLinalgHoisting.cpp
  TestLinalgTransforms.cpp
  TestPadFusion.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRArithmeticDialect
  MLIRArithmeticTransforms
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRGPUTransforms
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRLLVMToLLVMIRTranslation
  MLIRMemRefDialect
  MLIRParser
  MLIRPass
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRTensorDialect
  MLIRTensorTransforms
  MLIRTransformUtils
  MLIRVectorDialect
  MLIRVectorToSCF
  MLIRVectorTransforms
  )
