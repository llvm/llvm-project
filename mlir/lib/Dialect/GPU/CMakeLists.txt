if (MLIR_ENABLE_CUDA_CONVERSIONS)
  set(NVPTX_LIBS
    NVPTXCodeGen
    NVPTXDesc
    NVPTXInfo
  )
endif()

if (MLIR_ENABLE_ROCM_CONVERSIONS)
  set(AMDGPU_LIBS
    MCParser
    AMDGPUAsmParser
    AMDGPUCodeGen
    AMDGPUDesc
    AMDGPUInfo
  )
endif()

add_mlir_dialect_library(MLIRGPU
  IR/GPUDialect.cpp
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
  MLIRGPUOpsIncGen
  MLIRGPUOpInterfacesIncGen
  MLIRGPUPassIncGen
  MLIRParallelLoopMapperAttrGen
  MLIRParallelLoopMapperEnumsGen

  LINK_LIBS PUBLIC
  MLIRAsync
  MLIRDataLayoutInterfaces
  MLIRDLTI
  MLIRIR
  MLIRMemRef
  MLIRLLVMIR
  MLIRLLVMToLLVMIRTranslation
  MLIRSCF
  MLIRPass
  MLIRSideEffectInterfaces
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
  target_compile_definitions(obj.MLIRGPU
    PRIVATE
    MLIR_GPU_TO_CUBIN_PASS_ENABLE=1
  )

  # Add CUDA headers includes and the libcuda.so library.
  target_include_directories(obj.MLIRGPU
    PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  )

  find_library(CUDA_DRIVER_LIBRARY cuda)

  target_link_libraries(MLIRGPU
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

  # Configure ROCm support.
  if (NOT DEFINED ROCM_PATH)
    if (NOT DEFINED ENV{ROCM_PATH})
      set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to which ROCm has been installed")
    else()
      set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "Path to which ROCm has been installed")
    endif()
    set(HIP_PATH "${ROCM_PATH}/hip" CACHE PATH " Path to which HIP has been installed")
  endif()
  set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH})
  find_package(HIP)
  if (NOT HIP_FOUND)
    message(SEND_ERROR "Building mlir with ROCm support requires a working ROCm and HIP install")
  else()
    message(STATUS "ROCm HIP version: ${HIP_VERSION}")
  endif()

  target_compile_definitions(obj.MLIRGPU
    PRIVATE
    __HIP_PLATFORM_HCC__
    __ROCM_PATH__="${ROCM_PATH}"
    MLIR_GPU_TO_HSACO_PASS_ENABLE=1
  )

  target_include_directories(obj.MLIRGPU
    PRIVATE
    ${MLIR_SOURCE_DIR}/../lld/include
    ${HIP_PATH}/include
    ${ROCM_PATH}/include
  )

  target_link_libraries(MLIRGPU
    PRIVATE
    lldELF
    MLIRROCDLToLLVMIRTranslation
  )

  # Link lldELF also to libmlir.so. Create an alias that starts with LLVM
  # because LINK_COMPONENTS elements are implicitly prefixed with LLVM.
  add_library(LLVMAliasTolldELF ALIAS lldELF)
  set_property(GLOBAL APPEND PROPERTY MLIR_LLVM_LINK_COMPONENTS AliasTolldELF)

endif()
