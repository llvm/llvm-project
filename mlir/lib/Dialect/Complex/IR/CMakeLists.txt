add_mlir_dialect_library(MLIRComplexDialect
  ComplexOps.cpp
  ComplexDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Complex

  DEPENDS
  MLIRComplexOpsIncGen
  MLIRComplexAttributesIncGen

  # ComplexDialect.cpp includes mlir/Transforms/InliningUtils.h.
  HEADER_LIBS
  MLIRTransformUtils

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRDialect
  MLIRInferTypeOpInterface
  MLIRIR
  )
