set(LLVM_OPTIONAL_SOURCES
  ConvertKernelFuncToCubin.cpp
)

set(SOURCES
  ConvertLaunchFuncToCudaCalls.cpp
)

if (MLIR_CUDA_CONVERSIONS_ENABLED)
 list(APPEND SOURCES "ConvertKernelFuncToCubin.cpp")
  set(NVPTX_LIBS
    NVPTXCodeGen
    NVPTXDesc
    NVPTXInfo
  )
endif()

add_mlir_conversion_library(MLIRGPUtoCUDATransforms
  ${SOURCES}

  DEPENDS
  MLIRConversionPassIncGen

  LINK_COMPONENTS
  Core
  MC
  ${NVPTX_LIBS}
  )
target_link_libraries(MLIRGPUtoCUDATransforms
  PUBLIC
  MLIRGPU
  MLIRIR
  MLIRLLVMIR
  MLIRNVVMIR
  MLIRPass
  MLIRSupport
  MLIRTargetNVVMIR
)
