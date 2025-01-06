add_mlir_conversion_library(MLIRSPIRVAttrToLLVMConversion
  AttrToLLVMConverter.cpp

  DEPENDS
  MLIRSPIRVEnumsIncGen
)
