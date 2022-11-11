add_mlir_conversion_library(MLIRMathToFuncs
  MathToFuncs.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/MathToFuncs

  DEPENDS
  MLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRLLVMDialect
  MLIRMathDialect
  MLIRPass
  MLIRTransforms
  MLIRVectorDialect
  MLIRVectorUtils
  )
