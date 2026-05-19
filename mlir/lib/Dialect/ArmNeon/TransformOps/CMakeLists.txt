add_mlir_dialect_library(MLIRArmNeonVectorTransformOps
  ArmNeonVectorTransformOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/ArmNeon/TransformOps

  DEPENDS
  MLIRArmNeonVectorTransformOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRVectorDialect
  MLIRTransformDialect
  MLIRArmNeonDialect
  MLIRArmNeonTransforms
  )
