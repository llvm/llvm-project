if (MLIR_CUDA_CONVERSIONS_ENABLED)
  set(NVPTX_LIBS
    NVPTXCodeGen
    NVPTXDesc
    NVPTXInfo
  )
endif()

if (MLIR_ROCM_CONVERSIONS_ENABLED)
  set(AMDGPU_LIBS
    AMDGPUCodeGen
    AMDGPUDesc
    AMDGPUInfo
  )
endif()

add_mlir_conversion_library(MLIRGPUtoGPURuntimeTransforms
  ConvertLaunchFuncToRuntimeCalls.cpp
  ConvertKernelFuncToBlob.cpp

  DEPENDS
  MLIRConversionPassIncGen
  intrinsics_gen

  LINK_COMPONENTS
  Core
  MC
  ${AMDGPU_LIBS}
  ${NVPTX_LIBS}

  LINK_LIBS PUBLIC
  MLIRGPU
  MLIRIR
  MLIRLLVMIR
  MLIRPass
  MLIRSupport
)
