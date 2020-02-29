set(LLVM_OPTIONAL_SOURCES
  ConvertKernelFuncToCubin.cpp
)

set(SOURCES
  ConvertLaunchFuncToCudaCalls.cpp
)

if (MLIR_CUDA_CONVERSIONS_ENABLED)
 list(APPEND SOURCES "ConvertKernelFuncToCubin.cpp")
endif()

add_mlir_conversion_library(MLIRGPUtoCUDATransforms
  ${SOURCES}
  LINK_LIBS
  LLVMNVPTXCodeGen
  LLVMNVPTXDesc
  LLVMNVPTXInfo
  MLIRGPU
  MLIRLLVMIR
  MLIRNVVMIR
  MLIRPass
  MLIRTargetNVVMIR
  )
