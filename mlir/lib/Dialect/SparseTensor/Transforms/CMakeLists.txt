add_mlir_dialect_library(MLIRSparseTensorTransforms
  # Rewriting.
  BufferizableOpInterfaceImpl.cpp
  SparseBufferRewriting.cpp
  SparseGPUCodegen.cpp
  SparseReinterpretMap.cpp
  SparseStorageSpecifierToLLVM.cpp
  SparseTensorCodegen.cpp
  SparseTensorConversion.cpp
  SparseTensorPasses.cpp
  SparseTensorRewriting.cpp
  SparseVectorization.cpp
  Sparsification.cpp
  SparsificationAndBufferizationPass.cpp
  StageSparseOperations.cpp
  # Utilities.
  Utils/CodegenEnv.cpp
  Utils/CodegenUtils.cpp
  Utils/IterationGraphSorter.cpp
  Utils/LoopEmitter.cpp
  Utils/SparseTensorDescriptor.cpp

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
