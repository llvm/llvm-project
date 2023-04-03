add_mlir_dialect_library(MLIRSparseTensorTransforms
  BufferizableOpInterfaceImpl.cpp
  CodegenEnv.cpp
  CodegenUtils.cpp
  LoopEmitter.cpp
  SparseBufferRewriting.cpp
  SparseGPUCodegen.cpp
  SparseStorageSpecifierToLLVM.cpp
  SparseTensorCodegen.cpp
  SparseTensorConversion.cpp
  SparseTensorPasses.cpp
  SparseTensorRewriting.cpp
  SparseTensorStorageLayout.cpp
  SparseVectorization.cpp
  Sparsification.cpp
  SparsificationAndBufferizationPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/SparseTensor

  DEPENDS
  MLIRSparseTensorPassIncGen

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
  MLIRSparseTensorDialect
  MLIRSparseTensorEnums
  MLIRSparseTensorUtils
  MLIRTensorDialect
  MLIRTransforms
  MLIRVectorDialect
)
