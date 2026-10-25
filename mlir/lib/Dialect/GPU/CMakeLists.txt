add_mlir_library(MLIRGPUBase
  IR/GPUBase.cpp

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
  MLIRDeviceMappingInterfacesIncGen
  intrinsics_gen

  LINK_LIBS PUBLIC
  MLIRFunctionInterfaces
  MLIRIR
  MLIRSupport
  )

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
  MLIRParallelLoopMapperEnumsGen
  MLIRGPUDeviceMapperEnumsGen
  # GPU dialect sources include llvm/IR/Attributes.h.
  intrinsics_gen

  LINK_LIBS PUBLIC
  MLIRGPUBase
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
  Transforms/ParallelLoopMapper.cpp
  Transforms/ShuffleRewriter.cpp
  Transforms/SubgroupIdRewriter.cpp
  Transforms/SubgroupReduceLowering.cpp

  OBJECT

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/GPU

  DEPENDS
  MLIRGPUPassIncGen
  MLIRParallelLoopMapperEnumsGen

  LINK_LIBS PUBLIC
  MLIRAffineUtils
  MLIRArithDialect
  MLIRAsyncDialect
  MLIRBufferizationDialect
  MLIRDataLayoutInterfaces
  MLIRDialectUtils
  MLIRGPUDialect
  MLIRGPUUtils
  MLIRIR
  MLIRIndexDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
  MLIRSideEffectInterfaces
  MLIRSupport
  MLIRTransformUtils
  MLIRVectorDialect
  )

add_mlir_dialect_library(MLIRGPUNVVMTransforms
  Transforms/NVVM/NVVMAttachTarget.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/GPU/Transforms/NVVM

  DEPENDS
  MLIRGPUNVVMPassIncGen

  LINK_LIBS PUBLIC
  MLIRGPUDialect
  MLIRIR
  MLIRNVVMDialect
  MLIRPass
  MLIRSupport
  )

add_mlir_dialect_library(MLIRGPUROCDLTransforms
  Transforms/ROCDL/PromoteShuffleToAMDGPU.cpp
  Transforms/ROCDL/ROCDLAttachTarget.cpp
  Transforms/ROCDL/SubgroupReduceToDPP.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/GPU/Transforms/ROCDL

  DEPENDS
  MLIRGPUROCDLPassIncGen

  LINK_LIBS PUBLIC
  MLIRAMDGPUDialect
  MLIRArithDialect
  MLIRGPUDialect
  MLIRGPUUtils
  MLIRIR
  MLIRPass
  MLIRROCDLDialect
  MLIRSupport
  MLIRVectorDialect
  )

add_mlir_dialect_library(MLIRGPUSPIRVTransforms
  Transforms/SPIRV/SPIRVAttachTarget.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/GPU/Transforms/SPIRV

  DEPENDS
  MLIRGPUSPIRVPassIncGen

  LINK_LIBS PUBLIC
  MLIRGPUDialect
  MLIRIR
  MLIRPass
  MLIRSPIRVDialect
  MLIRSupport
  )

add_mlir_dialect_library(MLIRGPUXeVMTransforms
  Transforms/XeVM/XeVMAttachTarget.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/GPU/Transforms/XeVM

  DEPENDS
  MLIRGPUXeVMPassIncGen

  LINK_LIBS PUBLIC
  MLIRGPUDialect
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRXeVMDialect
  )

add_subdirectory(TransformOps)
add_subdirectory(Pipelines)
add_subdirectory(Utils)
