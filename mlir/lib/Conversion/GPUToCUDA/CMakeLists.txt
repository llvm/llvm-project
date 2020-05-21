set(LLVM_OPTIONAL_SOURCES
  ConvertKernelFuncToCubin.cpp
)

if (MLIR_CUDA_CONVERSIONS_ENABLED)
  set(NVPTX_LIBS
    MC
    NVPTXCodeGen
    NVPTXDesc
    NVPTXInfo
  )

  add_mlir_conversion_library(MLIRGPUtoCUDATransforms
    ConvertKernelFuncToCubin.cpp

    DEPENDS
    MLIRConversionPassIncGen
    intrinsics_gen

    LINK_COMPONENTS
    Core
    ${NVPTX_LIBS}

    LINK_LIBS PUBLIC
    MLIRGPU
    MLIRIR
    MLIRLLVMIR
    MLIRNVVMIR
    MLIRPass
    MLIRSupport
    MLIRTargetNVVMIR
  )
else()
  add_library(MLIRGPUtoCUDATransforms INTERFACE IMPORTED GLOBAL)
endif()
