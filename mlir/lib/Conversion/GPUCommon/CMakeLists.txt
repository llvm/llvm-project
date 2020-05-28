add_mlir_conversion_library(MLIRGPUtoGPURuntimeTransforms
  ConvertLaunchFuncToRuntimeCalls.cpp
  ConvertKernelFuncToBlob.cpp

  DEPENDS
  MLIRConversionPassIncGen
  intrinsics_gen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRGPU
  MLIRIR
  MLIRLLVMIR
  MLIRPass
  MLIRSupport
)
