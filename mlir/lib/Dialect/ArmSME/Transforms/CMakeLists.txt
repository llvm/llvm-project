add_mlir_dialect_library(MLIRArmSMETransforms
  EnableArmStreaming.cpp
  OuterProductFusion.cpp
  TileAllocation.cpp
  VectorLegalization.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/ArmSME/Transforms

  DEPENDS
  MLIRArmSMETransformsIncGen

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRArmSMEDialect
  MLIRFuncDialect
  MLIRVectorDialect
  MLIRIndexDialect
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRFuncTransforms
  )
