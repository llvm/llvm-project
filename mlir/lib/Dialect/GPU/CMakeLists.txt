if (MLIR_ENABLE_ROCM_CONVERSIONS)
  set(AMDGPU_LIBS
    IRReader
    IPO
    linker
    MCParser
    AMDGPUAsmParser
    AMDGPUCodeGen
    AMDGPUDesc
    AMDGPUInfo
    target
  )
endif()

add_mlir_dialect_library(MLIRGPUDialect
  IR/GPUDialect.cpp
  IR/InferIntRangeInterfaceImpls.cpp

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
  MLIRMemRefDialect
  MLIRSideEffectInterfaces
  MLIRSupport
  )

add_mlir_dialect_library(MLIRGPUTransforms
  Transforms/AllReduceLowering.cpp
  Transforms/AsyncRegionRewriter.cpp
  Transforms/BufferDeallocationOpInterfaceImpl.cpp
  Transforms/DecomposeMemrefs.cpp
  Transforms/EliminateBarriers.cpp
  Transforms/GlobalIdRewriter.cpp
  Transforms/KernelOutlining.cpp
  Transforms/MemoryPromotion.cpp
  Transforms/ModuleToBinary.cpp
  Transforms/NVVMAttachTarget.cpp
  Transforms/ParallelLoopMapper.cpp
  Transforms/ROCDLAttachTarget.cpp
  Transforms/SerializeToBlob.cpp
  Transforms/SerializeToHsaco.cpp
  Transforms/ShuffleRewriter.cpp
  Transforms/SPIRVAttachTarget.cpp
  Transforms/SubgroupReduceLowering.cpp
  Transforms/Utils.cpp
  
  OBJECT

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/GPU

  LINK_COMPONENTS
  Core
  MC
  Target
  ${AMDGPU_LIBS}

  DEPENDS
  MLIRGPUPassIncGen
  MLIRParallelLoopMapperEnumsGen

  LINK_LIBS PUBLIC
  MLIRAffineUtils
  MLIRArithDialect
  MLIRAsyncDialect
  MLIRBufferizationDialect
  MLIRBuiltinToLLVMIRTranslation
  MLIRDataLayoutInterfaces
  MLIRExecutionEngineUtils
  MLIRGPUDialect
  MLIRIR
  MLIRIndexDialect
  MLIRLLVMDialect
  MLIRGPUToLLVMIRTranslation
  MLIRLLVMToLLVMIRTranslation
  MLIRMemRefDialect
  MLIRNVVMTarget
  MLIRPass
  MLIRSCFDialect
  MLIRSideEffectInterfaces
  MLIRSPIRVTarget
  MLIRSupport
  MLIRROCDLTarget
  MLIRTransformUtils
  MLIRVectorDialect
  )

add_subdirectory(TransformOps)
add_subdirectory(Pipelines)

if(MLIR_ENABLE_ROCM_CONVERSIONS)
  if (NOT ("AMDGPU" IN_LIST LLVM_TARGETS_TO_BUILD))
    message(SEND_ERROR
      "Building mlir with ROCm support requires the AMDGPU backend")
  endif()

  set(DEFAULT_ROCM_PATH "/opt/rocm" CACHE PATH "Fallback path to search for ROCm installs")
  target_compile_definitions(obj.MLIRGPUTransforms
    PRIVATE
    __DEFAULT_ROCM_PATH__="${DEFAULT_ROCM_PATH}"
  )

  target_link_libraries(MLIRGPUTransforms
    PRIVATE
    MLIRROCDLToLLVMIRTranslation
  )
endif()
