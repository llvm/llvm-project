add_mlir_conversion_library(MLIRSCFToAffine
  RaiseToAffine.cpp
  AffineCFG.cpp
  Ops.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/SCFToAffine

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRAffineDialect
  MLIRLLVMDialect
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRTransforms
)
