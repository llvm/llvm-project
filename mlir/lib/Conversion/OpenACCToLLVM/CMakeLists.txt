add_mlir_conversion_library(MLIROpenACCToLLVM
  OpenACCToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/OpenACCToLLVM

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
  MLIRMemRefToLLVM
  MLIROpenACCDialect
  MLIRTransforms
  )
