add_mlir_conversion_library(MLIRLoopsToGPU
  LoopsToGPU.cpp
  LoopsToGPUPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/LoopsToGPU

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
