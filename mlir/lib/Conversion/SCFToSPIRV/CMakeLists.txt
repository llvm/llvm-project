add_mlir_conversion_library(MLIRSCFToSPIRV
  SCFToSPIRV.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/SCFToSPIRV

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRAffineOps
  MLIRAffineToStandard
  MLIRSPIRV
  MLIRIR
  MLIRLinalgOps
  MLIRPass
  MLIRStandardOps
  MLIRSupport
  MLIRTransforms
  )
