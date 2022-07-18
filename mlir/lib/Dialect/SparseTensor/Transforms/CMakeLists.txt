add_mlir_dialect_library(MLIRSparseTensorTransforms
  BufferizableOpInterfaceImpl.cpp
  CodegenUtils.cpp
  DenseBufferizationPass.cpp
  Sparsification.cpp
  SparseTensorConversion.cpp
  SparseTensorPasses.cpp
  SparseTensorRewriting.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/SparseTensor

  DEPENDS
  MLIRSparseTensorPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithmeticDialect
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
  MLIRSparseTensorDialect
  MLIRSparseTensorUtils
  MLIRTensorDialect
  MLIRTransforms
  MLIRVectorDialect
)
