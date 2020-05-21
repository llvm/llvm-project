set(SOURCES
  ConvertLaunchFuncToRuntimeCalls.cpp
)

add_mlir_conversion_library(MLIRGPUtoGPURuntimeTransforms
  ${SOURCES}

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
