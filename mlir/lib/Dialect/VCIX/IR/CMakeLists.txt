add_mlir_dialect_library(MLIRVCIXDialect
  VCIXDialect.cpp
  VCIXAttrs.cpp
  VCIXOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/VCIX

  DEPENDS
  MLIRVCIXIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
  MLIRSideEffectInterfaces
  )
