add_mlir_conversion_library(MLIRWasmRaise
  RaiseWasmMLIR.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/RaiseWasm

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRTransforms
  MLIRVectorDialect
  MLIRWasmSSADialect
  )
