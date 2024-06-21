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
  MLIRIR
  MLIRPass
  MLIRRewrite
  MLIRSPIRVConversion
  MLIRSPIRVDialect
  MLIRSupport
  MLIRTransformUtils
  )
