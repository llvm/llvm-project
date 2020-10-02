add_mlir_conversion_library(MLIRSCFToSPIRV
  SCFToSPIRV.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/SCFToSPIRV

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRAffine
  MLIRAffineToStandard
  MLIRSPIRV
  MLIRIR
  MLIRLinalg
  MLIRPass
  MLIRStandard
  MLIRSupport
  MLIRTransforms
  )
