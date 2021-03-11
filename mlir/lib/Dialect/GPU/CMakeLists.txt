if (MLIR_CUDA_CONVERSIONS_ENABLED)
  set(NVPTX_LIBS
    NVPTXCodeGen
    NVPTXDesc
    NVPTXInfo
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

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/GPU

  LINK_COMPONENTS
  Core
  MC
  ${NVPTX_LIBS}

  DEPENDS
  MLIRGPUOpsIncGen
  MLIRGPUOpInterfacesIncGen
  MLIRGPUPassIncGen
  MLIRParallelLoopMapperAttrGen
  MLIRParallelLoopMapperEnumsGen

  LINK_LIBS PUBLIC
  MLIRAsync
  MLIREDSC
  MLIRIR
  MLIRLLVMIR
  MLIRLLVMToLLVMIRTranslation
  MLIRSCF
  MLIRPass
  MLIRSideEffectInterfaces
  MLIRStandard
  MLIRSupport
  MLIRTransformUtils
  )

if(MLIR_CUDA_RUNNER_ENABLED)
  if(NOT MLIR_CUDA_CONVERSIONS_ENABLED)
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
