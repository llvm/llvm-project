# Keep the complete public library as an umbrella over its components.
add_mlir_library(MLIRGPUCore
  OBJECT
  IR/GPUDialect.cpp
  IR/InferIntRangeInterfaceImpls.cpp

  PARTIAL_SOURCES_INTENDED

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/GPU

  DEPENDS
  MLIRGPUOpsIncGen
  MLIRGPUOpsAttributesIncGen
  MLIRGPUOpsEnumsGen
  MLIRGPUOpInterfacesIncGen
  MLIRGPUCompilationAttrInterfacesIncGen
  MLIRParallelLoopMapperEnumsGen
  MLIRGPUDeviceMapperEnumsGen
  # GPU dialect sources include llvm/IR/Attributes.h.
  intrinsics_gen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRDLTIDialect
  MLIRControlFlowInterfaces
  MLIRFunctionInterfaces
  MLIRInferIntRangeInterface
  MLIRIRCore
  MLIRLoopLikeInterface
  MLIRMathOps
  MLIRMemRefCore
  MLIRSideEffectInterfaces
  MLIRSupportCore
  )

add_mlir_dialect_library(MLIRGPUDialect
  IR/ValueBoundsOpInterfaceImpl.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/GPU

  DEPENDS
  MLIRGPUOpsIncGen
  MLIRGPUOpsAttributesIncGen
  MLIRGPUOpsEnumsGen
  MLIRGPUOpInterfacesIncGen
  MLIRGPUCompilationAttrInterfacesIncGen
  MLIRParallelLoopMapperEnumsGen
  MLIRGPUDeviceMapperEnumsGen
  # GPU dialect sources include llvm/IR/Attributes.h.
  intrinsics_gen

  LINK_LIBS PUBLIC
  MLIRGPUCore
  MLIRArithDialect
  MLIRDLTIDialect
  MLIRControlFlowInterfaces
  MLIRFunctionInterfaces
  MLIRInferIntRangeInterface
  MLIRIR
  MLIRLoopLikeInterface
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRSideEffectInterfaces
  MLIRSupport
  )

# Preserve the complete public archive. The component objects are compiled
# once; libMLIR still collects each object's defining library exactly once.
if(NOT BUILD_SHARED_LIBS)
  target_sources(MLIRGPUDialect PRIVATE $<TARGET_OBJECTS:obj.MLIRGPUCore>)
endif()

add_mlir_dialect_library(MLIRGPUTransforms
  Transforms/AllReduceLowering.cpp
  Transforms/AsyncRegionRewriter.cpp
  Transforms/BufferDeallocationOpInterfaceImpl.cpp
  Transforms/DecomposeMemRefs.cpp
  Transforms/EliminateBarriers.cpp
  Transforms/GlobalIdRewriter.cpp
  Transforms/KernelOutlining.cpp
  Transforms/IndexedAccessOpInterfaceImpl.cpp
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

  LINK_LIBS PUBLIC
  MLIRAMDGPUDialect
  MLIRAffineUtils
  MLIRArithDialect
  MLIRAsyncDialect
  MLIRBufferizationDialect
  MLIRDataLayoutInterfaces
  MLIRDialectUtils
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
