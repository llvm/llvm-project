# Exclude tests from libMLIR.so
add_mlir_library(MLIRLinalgTestPasses
  TestDataLayoutPropagation.cpp
  TestLinalgDecomposeOps.cpp
  TestLinalgDropUnitDims.cpp
  TestLinalgElementwiseFusion.cpp
  TestLinalgFusionTransforms.cpp
  TestLinalgTransforms.cpp
  TestPadFusion.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRArithDialect
  MLIRArithTransforms
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
