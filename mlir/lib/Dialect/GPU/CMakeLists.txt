if ("NVPTX" IN_LIST LLVM_TARGETS_TO_BUILD)
  set(NVPTX_LIBS
    NVPTXCodeGen
    NVPTXDesc
    NVPTXInfo
  )
endif()

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
  Transforms/GlobalIdRewriter.cpp
  Transforms/KernelOutlining.cpp
  Transforms/MemoryPromotion.cpp
  Transforms/ModuleToBinary.cpp
  Transforms/NVVMAttachTarget.cpp
  Transforms/ParallelLoopMapper.cpp
  Transforms/SerializeToBlob.cpp
  Transforms/SerializeToCubin.cpp
  Transforms/SerializeToHsaco.cpp
  Transforms/ShuffleRewriter.cpp
  Transforms/SPIRVAttachTarget.cpp
  Transforms/ROCDLAttachTarget.cpp
  Transforms/EliminateBarriers.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/GPU

  LINK_COMPONENTS
  Core
  MC
  Target
  ${NVPTX_LIBS}
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

if(MLIR_ENABLE_CUDA_RUNNER)
  if(NOT MLIR_ENABLE_CUDA_CONVERSIONS)
    message(SEND_ERROR
      "Building mlir with cuda support requires the NVPTX backend")
  endif()

  # Configure CUDA language support. Using check_language first allows us to
  # give a custom error message.
  include(CheckLanguage)
  check_language(CUDA)
  if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
  else()
    message(SEND_ERROR
      "Building mlir with cuda support requires a working CUDA install")
  endif()

  # Enable gpu-to-cubin pass.
  target_compile_definitions(obj.MLIRGPUTransforms
    PRIVATE
    MLIR_GPU_TO_CUBIN_PASS_ENABLE=1
  )

  # Add CUDA headers includes and the libcuda.so library.
  target_include_directories(obj.MLIRGPUTransforms
    PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  )

  # Add link path for the cuda driver library.
  find_library(CUDA_DRIVER_LIBRARY cuda HINTS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES} REQUIRED)
  get_filename_component(CUDA_DRIVER_LIBRARY_PATH "${CUDA_DRIVER_LIBRARY}" DIRECTORY)
  target_link_directories(MLIRGPUTransforms PRIVATE ${CUDA_DRIVER_LIBRARY_PATH})

  target_link_libraries(MLIRGPUTransforms
    PRIVATE
    MLIRNVVMToLLVMIRTranslation
    cuda
  )

endif()

if(MLIR_ENABLE_ROCM_CONVERSIONS)
  if (NOT ("AMDGPU" IN_LIST LLVM_TARGETS_TO_BUILD))
    message(SEND_ERROR
      "Building mlir with ROCm support requires the AMDGPU backend")
  endif()

  set(DEFAULT_ROCM_PATH "/opt/rocm" CACHE PATH "Fallback path to search for ROCm installs")
  target_compile_definitions(obj.MLIRGPUTransforms
    PRIVATE
    __DEFAULT_ROCM_PATH__="${DEFAULT_ROCM_PATH}"
    MLIR_GPU_TO_HSACO_PASS_ENABLE=1
  )

  target_link_libraries(MLIRGPUTransforms
    PRIVATE
    MLIRROCDLToLLVMIRTranslation
  )
endif()
