add_mlir_conversion_library(MLIRSCFToSPIRV
  SCFToSPIRV.cpp
  SCFToSPIRVPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/SCFToSPIRV

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithmeticToSPIRV
  MLIRFuncToSPIRV
  MLIRMemRefToSPIRV
  MLIRSPIRV
  MLIRSPIRVConversion
  MLIRIR
  MLIRPass
  MLIRSCF
  MLIRSupport
  MLIRTransforms
  )
