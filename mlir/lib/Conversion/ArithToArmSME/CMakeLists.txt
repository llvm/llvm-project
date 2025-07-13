add_mlir_conversion_library(MLIRArithToArmSME
  ArithToArmSME.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/ArithToArmSME

  DEPENDS
  MLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRArmSMEDialect
  MLIRArithDialect
  MLIRPass
  MLIRTransforms
  )
