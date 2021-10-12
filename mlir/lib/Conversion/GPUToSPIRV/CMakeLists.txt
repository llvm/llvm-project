add_mlir_conversion_library(MLIRGPUToSPIRV
  GPUToSPIRV.cpp
  GPUToSPIRVPass.cpp

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithmeticToSPIRV
  MLIRGPUOps
  MLIRIR
  MLIRPass
  MLIRSCFToSPIRV
  MLIRSPIRV
  MLIRSPIRVConversion
  MLIRStandardToSPIRV
  MLIRSupport
  MLIRTransforms
  )
