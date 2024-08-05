set(LLVM_OPTIONAL_SOURCES
  ConvertToSPIRVPass.cpp
  ToSPIRVInterface.cpp
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
  MLIRSCFToSPIRV
  MLIRSPIRVConversion
  MLIRSPIRVDialect
  MLIRSPIRVTransforms
  MLIRSupport
  MLIRTransformUtils
  )

add_mlir_conversion_library(MLIRConvertToSPIRVInterface
  ToSPIRVInterface.cpp

  DEPENDS

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
)
