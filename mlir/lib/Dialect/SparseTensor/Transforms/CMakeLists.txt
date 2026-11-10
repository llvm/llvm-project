set(LLVM_OPTIONAL_SOURCES
  BufferizableOpInterfaceImpl.cpp
  )

add_mlir_dialect_library(MLIRSparseTensorTransforms
  # Rewriting.
  SparseAssembler.cpp
  SparseBufferRewriting.cpp
  SparseGPUCodegen.cpp
  SparseIterationToScf.cpp
  SparseReinterpretMap.cpp
  SparseStorageSpecifierToLLVM.cpp
  SparseSpaceCollapse.cpp
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
  Utils/SparseTensorIterator.cpp

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
  MLIRLLVMCommonConversion
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

add_mlir_dialect_library(MLIRSparseTensorBufferizableOpInterfaceImpl
  BufferizableOpInterfaceImpl.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/SparseTensor

  LINK_LIBS PUBLIC
  MLIRBufferizationDialect
  MLIRIR
  MLIRSparseTensorDialect
)
