set(LLVM_TARGET_DEFINITIONS GPUToSPIRV.td)
mlir_tablegen(GPUToSPIRV.cpp.inc -gen-rewriters)
add_public_tablegen_target(MLIRGPUToSPIRVIncGen)

add_mlir_conversion_library(MLIRGPUToSPIRV
  GPUToSPIRV.cpp
  GPUToSPIRVPass.cpp

  DEPENDS
  MLIRConversionPassIncGen
  MLIRGPUToSPIRVIncGen

  LINK_LIBS PUBLIC
  MLIRGPU
  MLIRIR
  MLIRPass
  MLIRSCFToSPIRV
  MLIRSPIRV
  MLIRSPIRVConversion
  MLIRStandard
  MLIRStandardToSPIRV
  MLIRSupport
  MLIRTransforms
  )
