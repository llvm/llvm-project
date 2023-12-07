add_mlir_dialect_library(MLIRArmSMETransforms
  ArmSMETypeConverter.cpp
  EnableArmStreaming.cpp
  LegalizeForLLVMExport.cpp
  TileAllocation.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/ArmSME/Transforms

  DEPENDS
  MLIRArmSMETransformsIncGen

  LINK_LIBS PUBLIC
  MLIRArmSMEDialect
  MLIRArmSMEUtils
  MLIRFuncDialect
  MLIRLLVMCommonConversion
  MLIRVectorDialect
  MLIRSCFDialect
  MLIRPass
  )
