add_mlir_conversion_library(MLIRMemRefToSPIRV
  MapMemRefStorageClassPass.cpp
  MemRefToSPIRV.cpp
  MemRefToSPIRVPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/SPIRV
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/IR

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRFunctionInterfaces
  MLIRIR
  MLIRMemRefDialect
  MLIRPass
  MLIRSPIRVDialect
  MLIRSPIRVConversion
  MLIRSupport
  MLIRTransformUtils
  )

