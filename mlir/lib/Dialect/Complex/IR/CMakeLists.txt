add_mlir_dialect_library(MLIRComplexDialect
  ComplexOps.cpp
  ComplexDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Complex

  DEPENDS
  MLIRComplexOpsIncGen
  MLIRComplexAttributesIncGen

  LINK_LIBS PUBLIC
  MLIRArithAttrToLLVMConversion
  MLIRArithDialect
  MLIRDialect
  MLIRInferTypeOpInterface
  MLIRIR
  )
