add_mlir_conversion_library(MLIRControlFlowToSCF
  ControlFlowToSCF.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/ControlFlowToSCF

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRSCFDialect
  MLIRUBDialect
  MLIRPass
  MLIRTransformUtils
)
