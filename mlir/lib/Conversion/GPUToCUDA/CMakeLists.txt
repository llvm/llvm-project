set(LLVM_OPTIONAL_SOURCES
  ConvertKernelFuncToCubin.cpp
)

set(SOURCES
  ConvertLaunchFuncToCudaCalls.cpp
)

if (MLIR_CUDA_CONVERSIONS_ENABLED)
 list(APPEND SOURCES "ConvertKernelFuncToCubin.cpp")
  set(NVPTX_LIBS
    LLVMNVPTXCodeGen
    LLVMNVPTXDesc
    LLVMNVPTXInfo
  )
endif()

add_mlir_conversion_library(MLIRGPUtoCUDATransforms ${SOURCES})
target_link_libraries(MLIRGPUtoCUDATransforms
  PUBLIC
  ${NVPTX_LIBS}
  LLVMCore
  LLVMMC
  LLVMSupport
  MLIRGPU
  MLIRIR
  MLIRLLVMIR
  MLIRNVVMIR
  MLIRPass
  MLIRSupport
  MLIRTargetNVVMIR
)
