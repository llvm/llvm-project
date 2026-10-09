add_mlir_conversion_library(MLIRSPIRVAttrToLLVMConversion
  AttrToLLVMConverter.cpp

  DEPENDS
  MLIRSPIRVEnumsIncGen

  # AttrToLLVMConverter.h includes the generated SPIRVEnums.h.
  HEADER_LIBS
  MLIRSPIRVDialect
)
