add_mlir_conversion_library(MLIRMathToLibm
  MathToLibm.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/MathToLibm

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRDialectUtils
  MLIRFuncDialect
  MLIRLLVMDialect
  MLIRMathDialect
  MLIRPass
  MLIRTransformUtils
  MLIRVectorDialect
  MLIRVectorUtils
  )
