add_mlir_conversion_library(MLIRSCFToGPU
  SCFToGPU.cpp
  SCFToGPUPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/SCFToGPU

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRAffineOps
  MLIRAffineToStandard
  MLIRGPU
  MLIRIR
  MLIRLinalgOps
  MLIRPass
  MLIRStandardOps
  MLIRSupport
  MLIRTransforms
  )
