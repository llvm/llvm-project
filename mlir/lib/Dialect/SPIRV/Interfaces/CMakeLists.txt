add_mlir_library(MLIRSPIRVImageInterfaces
  SPIRVImageInterfaces.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/SPIRV

  DEPENDS
  MLIRSPIRVImageInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)
