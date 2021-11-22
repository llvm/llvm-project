if (MLIR_ENABLE_CUDA_CONVERSIONS)
  set(NVPTX_LIBS
    NVPTXCodeGen
    NVPTXDesc
    NVPTXInfo
  )
endif()

if (MLIR_ENABLE_ROCM_CONVERSIONS)
  set(AMDGPU_LIBS
    IRReader
    linker
    MCParser
    AMDGPUAsmParser
    AMDGPUCodeGen
    AMDGPUDesc
    AMDGPUInfo
    target
  )
endif()

add_mlir_dialect_library(MLIRGPUOps
  IR/GPUDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/GPU

  DEPENDS
  MLIRGPUOpsIncGen
  MLIRGPUOpsEnumsGen
  MLIRGPUOpInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRArithmetic
  MLIRDLTI
  MLIRIR
  MLIRMemRef
  MLIRSideEffectInterfaces
  MLIRSupport
  MLIRLLVMIR
  )

add_mlir_dialect_library(MLIRGPUTransforms
  Transforms/AllReduceLowering.cpp
  Transforms/AsyncRegionRewriter.cpp
  Transforms/KernelOutlining.cpp
  Transforms/MemoryPromotion.cpp
  Transforms/ParallelLoopMapper.cpp
  Transforms/SerializeToBlob.cpp
  Transforms/SerializeToCubin.cpp
  Transforms/SerializeToHsaco.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/GPU

  LINK_COMPONENTS
  Core
  MC
  ${NVPTX_LIBS}
  ${AMDGPU_LIBS}

  DEPENDS
  MLIRGPUPassIncGen
  MLIRParallelLoopMapperAttrGen
  MLIRParallelLoopMapperEnumsGen

  LINK_LIBS PUBLIC
  MLIRArithmetic
  MLIRAsync
  MLIRDataLayoutInterfaces
  MLIRGPUOps
  MLIRIR
  MLIRLLVMIR
  MLIRLLVMToLLVMIRTranslation
  MLIRMemRef
  MLIRPass
  MLIRSCF
  MLIRStandard
  MLIRSupport
  MLIRTransformUtils
  )

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

  find_library(CUDA_DRIVER_LIBRARY cuda)

  target_link_libraries(MLIRGPUTransforms
    PRIVATE
    MLIRNVVMToLLVMIRTranslation
    ${CUDA_DRIVER_LIBRARY}
  )

endif()

if(MLIR_ENABLE_ROCM_RUNNER)
  if (NOT ("AMDGPU" IN_LIST LLVM_TARGETS_TO_BUILD))
    message(SEND_ERROR
      "Building mlir with ROCm support requires the AMDGPU backend")
  endif()

  # Ensure lld is enabled.
  if (NOT "lld" IN_LIST LLVM_ENABLE_PROJECTS)
    message(SEND_ERROR "lld is not enabled. Please revise LLVM_ENABLE_PROJECTS")
  endif()

  set(DEFAULT_ROCM_PATH "/opt/rocm" CACHE PATH "Fallback path to search for ROCm installs")
  target_compile_definitions(obj.MLIRGPUTransforms
    PRIVATE
    __DEFAULT_ROCM_PATH__="${DEFAULT_ROCM_PATH}"
    MLIR_GPU_TO_HSACO_PASS_ENABLE=1
  )

  target_include_directories(obj.MLIRGPUTransforms
    PRIVATE
    ${MLIR_SOURCE_DIR}/../lld/include
  )

  target_link_libraries(MLIRGPUOps
    PRIVATE
    lldELF
    MLIRExecutionEngine
    MLIRROCDLToLLVMIRTranslation
  )

  # Link lldELF also to libmlir.so. Create an alias that starts with LLVM
  # because LINK_COMPONENTS elements are implicitly prefixed with LLVM.
  add_library(LLVMAliasTolldELF ALIAS lldELF)
  set_property(GLOBAL APPEND PROPERTY MLIR_LLVM_LINK_COMPONENTS AliasTolldELF)

endif()
