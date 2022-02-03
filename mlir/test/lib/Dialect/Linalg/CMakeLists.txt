# Exclude tests from libMLIR.so
add_mlir_library(MLIRLinalgTestPasses
  TestComprehensiveBufferize.cpp
  TestLinalgCodegenStrategy.cpp
  TestLinalgDistribution.cpp
  TestLinalgElementwiseFusion.cpp
  TestLinalgFusionTransforms.cpp
  TestLinalgHoisting.cpp
  TestLinalgTransforms.cpp
  TestPadFusion.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  MLIRAffine
  MLIRAffineBufferizableOpInterfaceImpl
  MLIRArithmetic
  MLIRArithmeticTransforms
  MLIRBufferization
  MLIRBufferizationTransforms
  MLIRGPUTransforms
  MLIRLinalg
  MLIRLinalgBufferizableOpInterfaceImpl
  MLIRLinalgTransforms
  MLIRLLVMToLLVMIRTranslation
  MLIRMemRef
  MLIRPass
  MLIRSCF
  MLIRSCFTransforms
  MLIRStandard
  MLIRStandardOpsTransforms
  MLIRTensor
  MLIRTensorTransforms
  MLIRTransformUtils
  MLIRVector
  MLIRVectorToSCF
  MLIRVectorTransforms
  )
