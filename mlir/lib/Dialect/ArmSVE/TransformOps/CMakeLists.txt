add_mlir_dialect_library(MLIRArmSVEVectorTransformOps
  ArmSVEVectorTransformOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/ArmSVE/TransformOps

  DEPENDS
  MLIRArmSVEVectorTransformOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRVectorDialect
  MLIRTransformDialect
  MLIRArmSVEDialect
  MLIRArmSVETransforms
  )
