# Exclude tests from libMLIR.so
add_mlir_library(MLIRLinalgTestPasses
  TestComprehensiveBufferize.cpp
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
  MLIRAffineBufferizableOpInterfaceImpl
  MLIRArithBufferizableOpInterfaceImpl
  MLIRArithmetic
  MLIRBufferizableOpInterface
  MLIRComprehensiveBufferize
  MLIRGPUTransforms
  MLIRLinalg
  MLIRLinalgBufferizableOpInterfaceImpl
  MLIRLinalgTransforms
  MLIRLLVMToLLVMIRTranslation
  MLIRMemRef
  MLIRPass
  MLIRSCF
  MLIRSCFBufferizableOpInterfaceImpl
  MLIRStandard
  MLIRTensor
  MLIRTensorBufferizableOpInterfaceImpl
  MLIRTransformUtils
  MLIRVector
  MLIRVectorBufferizableOpInterfaceImpl
  MLIRVectorToSCF
  )
