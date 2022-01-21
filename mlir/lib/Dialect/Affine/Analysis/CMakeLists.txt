add_mlir_dialect_library(MLIRAffineAnalysis
  AffineAnalysis.cpp
  AffineStructures.cpp
  LoopAnalysis.cpp
  NestedMatcher.cpp
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Affine

  DEPENDS
  mlir-headers

  LINK_LIBS PUBLIC
  MLIRAffine
  MLIRAnalysis
  MLIRCallInterfaces
  MLIRControlFlowInterfaces
  MLIRInferTypeOpInterface
  MLIRPresburger
  MLIRSCF
  )
