add_mlir_conversion_library(MLIRSCFToGPU
  SCFToGPU.cpp
  SCFToGPUPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/SCFToGPU

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRAffineToStandard
  MLIRArithDialect
  MLIRComplexDialect
  MLIRGPUTransforms
  MLIRIR
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRSupport
  MLIRTransforms
  )
