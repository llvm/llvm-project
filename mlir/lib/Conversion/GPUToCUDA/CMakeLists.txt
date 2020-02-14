set(LLVM_OPTIONAL_SOURCES
  ConvertKernelFuncToCubin.cpp
)

set(SOURCES
  ConvertLaunchFuncToCudaCalls.cpp
)

if (MLIR_CUDA_CONVERSIONS_ENABLED)
 list(APPEND SOURCES "ConvertKernelFuncToCubin.cpp")
endif()

add_llvm_library(MLIRGPUtoCUDATransforms ${SOURCES})
target_link_libraries(MLIRGPUtoCUDATransforms
  MLIRGPU
  MLIRLLVMIR
  MLIRNVVMIR
  MLIRPass
  MLIRTargetNVVMIR
)
