add_mlir_dialect_library(MLIRGPUDialect
  IR/GPUDialect.cpp
  IR/InferIntRangeInterfaceImpls.cpp
  IR/ValueBoundsOpInterfaceImpl.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/GPU

  DEPENDS
  MLIRGPUOpsIncGen
  MLIRGPUOpsAttributesIncGen
  MLIRGPUOpsEnumsGen
  MLIRGPUOpInterfacesIncGen
  MLIRGPUCompilationAttrInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRDLTIDialect
  MLIRControlFlowInterfaces
  MLIRFunctionInterfaces
  MLIRInferIntRangeInterface
  MLIRIR
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRSideEffectInterfaces
  MLIRSupport
  )

add_mlir_dialect_library(MLIRGPUTransforms
  Transforms/AllReduceLowering.cpp
  Transforms/AsyncRegionRewriter.cpp
  Transforms/BufferDeallocationOpInterfaceImpl.cpp
  Transforms/DecomposeMemRefs.cpp
  Transforms/EliminateBarriers.cpp
  Transforms/GlobalIdRewriter.cpp
  Transforms/KernelOutlining.cpp
  Transforms/MemoryPromotion.cpp
  Transforms/ModuleToBinary.cpp
  Transforms/NVVMAttachTarget.cpp
  Transforms/ParallelLoopMapper.cpp
  Transforms/PromoteShuffleToAMDGPU.cpp
  Transforms/ROCDLAttachTarget.cpp
  Transforms/SPIRVAttachTarget.cpp
  Transforms/ShuffleRewriter.cpp
  Transforms/SubgroupIdRewriter.cpp
  Transforms/SubgroupReduceLowering.cpp
  Transforms/XeVMAttachTarget.cpp

  OBJECT

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/GPU

  DEPENDS
  MLIRGPUPassIncGen
  MLIRParallelLoopMapperEnumsGen

  LINK_LIBS PUBLIC
  MLIRAMDGPUDialect
  MLIRAffineUtils
  MLIRArithDialect
  MLIRAsyncDialect
  MLIRBufferizationDialect
  MLIRDataLayoutInterfaces
  MLIRExecutionEngineUtils
  MLIRGPUDialect
  MLIRGPUUtils
  MLIRIR
  MLIRIndexDialect
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIRNVVMTarget
  MLIRPass
  MLIRROCDLDialect
  MLIRROCDLTarget
  MLIRSCFDialect
  MLIRSPIRVTarget
  MLIRSideEffectInterfaces
  MLIRSupport
  MLIRTransformUtils
  MLIRVectorDialect
  MLIRXeVMDialect
  )

add_subdirectory(TransformOps)
add_subdirectory(Pipelines)
add_subdirectory(Utils)
