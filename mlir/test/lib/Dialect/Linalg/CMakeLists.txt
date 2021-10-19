# Exclude tests from libMLIR.so
add_mlir_library(MLIRLinalgTestPasses
  TestConvVectorization.cpp
  TestLinalgCodegenStrategy.cpp
  TestLinalgDistribution.cpp
  TestLinalgElementwiseFusion.cpp
  TestLinalgFusionTransforms.cpp
  TestLinalgHoisting.cpp
  TestLinalgTransforms.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  MLIRAffine
  MLIRGPUTransforms
  MLIRLinalg
  MLIRLinalgTransforms
  MLIRLLVMToLLVMIRTranslation
  MLIRPass
  MLIRStandard
  MLIRTransformUtils
  MLIRVector
  MLIRVectorToSCF
  )
