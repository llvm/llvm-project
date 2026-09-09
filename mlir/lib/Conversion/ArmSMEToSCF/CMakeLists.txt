add_mlir_conversion_library(MLIRArmSMEToSCF
  ArmSMEToSCF.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/ArmSMEToSCF

  LINK_LIBS PUBLIC
  MLIRArmSMEDialect
  MLIRTransforms
  )
