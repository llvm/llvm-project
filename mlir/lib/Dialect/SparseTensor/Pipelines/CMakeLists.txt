add_mlir_dialect_library(MLIRSparseTensorPipelines
  SparseTensorPipelines.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/SparseTensor

  LINK_LIBS PUBLIC
  MLIRArithTransforms
  MLIRAffineToStandard
  MLIRBufferizationTransforms
  MLIRComplexToLLVM
  MLIRComplexToLibm
  MLIRComplexToStandard
  MLIRFuncTransforms
  MLIRGPUToNVVMTransforms
  MLIRGPUTransforms
  MLIRLinalgTransforms
  MLIRMathToLibm
  MLIRMathToLLVM
  MLIRMemRefToLLVM
  MLIRPass
  MLIRReconcileUnrealizedCasts
  MLIRSCFToControlFlow
  MLIRSparseTensorDialect
  MLIRSparseTensorTransforms
  MLIRTensorTransforms
  MLIRVectorToLLVM
  MLIRVectorTransforms
)

if(MLIR_ENABLE_CUDA_RUNNER)
   # Enable gpu-to-cubin pass.
  target_compile_definitions(obj.MLIRSparseTensorPipelines
    PRIVATE
    MLIR_GPU_TO_CUBIN_PASS_ENABLE=1
  )
endif()
