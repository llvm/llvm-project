set(LLVM_OPTIONAL_SOURCES
  ConvertToSPIRVPass.cpp
)

add_mlir_conversion_library(MLIRConvertToSPIRVPass
  ConvertToSPIRVPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/ConvertToSPIRV

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithToSPIRV
  MLIRArithTransforms
  MLIRFuncToSPIRV
  MLIRIndexToSPIRV
  MLIRIR
  MLIRPass
  MLIRRewrite
  MLIRSCFToSPIRV
  MLIRSPIRVConversion
  MLIRSPIRVDialect
  MLIRSPIRVTransforms
  MLIRSupport
  MLIRTransforms
  MLIRTransformUtils
  MLIRUBToSPIRV
  MLIRVectorToSPIRV
  MLIRVectorTransforms
  )
