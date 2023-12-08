add_mlir_conversion_library(MLIRVectorToArmSME
  VectorToArmSME.cpp
  VectorToArmSMEPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/VectorToArmSME

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArmSMEDialect
  MLIRLLVMCommonConversion
  )
