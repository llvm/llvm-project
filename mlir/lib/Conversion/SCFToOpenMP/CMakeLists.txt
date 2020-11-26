add_mlir_conversion_library(MLIRSCFToOpenMP
  SCFToOpenMP.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/SCFToStandard

  DEPENDS
  MLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIROpenMP
  MLIRSCF
  MLIRTransforms
  )
