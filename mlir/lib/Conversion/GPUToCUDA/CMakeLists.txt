set(LLVM_OPTIONAL_SOURCES
  ConvertKernelFuncToCubin.cpp
)

set(SOURCES
  ConvertLaunchFuncToCudaCalls.cpp
)

if (MLIR_CUDA_CONVERSIONS_ENABLED)
 list(APPEND SOURCES "ConvertKernelFuncToCubin.cpp")
endif()

add_mlir_conversion_library(MLIRGPUtoCUDATransforms ${SOURCES})
target_link_libraries(MLIRGPUtoCUDATransforms
  PUBLIC
  LLVMNVPTXCodeGen
  LLVMNVPTXDesc
  LLVMNVPTXInfo
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
