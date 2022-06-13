add_mlir_conversion_library(MLIRGPUToSPIRV
  GPUToSPIRV.cpp
  GPUToSPIRVPass.cpp

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithmeticToSPIRV
  MLIRGPUOps
  MLIRFuncToSPIRV
  MLIRIR
  MLIRPass
  MLIRSCFToSPIRV
  MLIRSPIRVDialect
  MLIRSPIRVConversion
  MLIRSupport
  MLIRTransforms
  )
