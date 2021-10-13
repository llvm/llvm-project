add_mlir_conversion_library(MLIROpenACCToSCF
  OpenACCToSCF.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/OpenACCToSCF

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithmetic
  MLIRIR
  MLIROpenACC
  MLIRSCF
  MLIRTransforms
  )
