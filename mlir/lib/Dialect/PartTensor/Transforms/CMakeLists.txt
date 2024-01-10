add_mlir_dialect_library(MLIRPartTensorTransforms
  PartTensorConversion.cpp
  PartTensorPasses.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/PartTensor

  DEPENDS
  MLIRPartTensorPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRComplexDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRSCFUtils
  MLIRPartTensor
  MLIRTensorDialect
  MLIRTransforms
  MLIRVectorDialect
)
