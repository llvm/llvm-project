add_mlir_dialect_library(MLIRMLProgramDialect
  MLProgramOps.cpp
  MLProgramDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/MLProgram

  DEPENDS
  MLIRMLProgramOpsIncGen
  MLIRMLProgramAttributesIncGen
  MLIRMLProgramTypesIncGen

  LINK_LIBS PUBLIC
  MLIRDialect
  MLIRControlFlowInterfaces
  MLIRFunctionInterfaces
  MLIRInferTypeOpInterface
  MLIRTransforms
  MLIRIR
  )
