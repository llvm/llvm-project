# Exclude tests from libMLIR.so
add_mlir_library(MLIRLinalgTestPasses
  TestDataLayoutPropagation.cpp
  TestLinalgDecomposeOps.cpp
  TestLinalgDropUnitDims.cpp
  TestLinalgElementwiseFusion.cpp
  TestLinalgFusionTransforms.cpp
  TestLinalgRankReduceContractionOps.cpp
  TestLinalgTransforms.cpp
  TestPadFusion.cpp

  EXCLUDE_FROM_LIBMLIR
  )
mlir_target_link_libraries(MLIRLinalgTestPasses PUBLIC
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
