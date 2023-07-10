add_mlir_dialect_library(MLIRArmSMETransforms
  EnableArmStreaming.cpp
  LegalizeForLLVMExport.cpp
  LowerVectorOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/ArmSME/Transforms

  DEPENDS
  MLIRArmSMETransformsIncGen

  LINK_LIBS PUBLIC
  MLIRArmSMEDialect
  MLIRFuncDialect
  MLIRLLVMCommonConversion
  MLIRVectorDialect
  MLIRSCFDialect
  MLIRPass
  )
