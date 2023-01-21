set(LLVM_TARGET_DEFINITIONS ArithCanonicalization.td)
mlir_tablegen(ArithCanonicalization.inc -gen-rewriters)
add_public_tablegen_target(MLIRArithCanonicalizationIncGen)

add_mlir_dialect_library(MLIRArithDialect
  ArithOps.cpp
  ArithDialect.cpp
  InferIntRangeInterfaceImpls.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Arith

  DEPENDS
  MLIRArithOpsIncGen
  MLIRArithOpsInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRDialect
  MLIRInferIntRangeCommon
  MLIRInferIntRangeInterface
  MLIRInferTypeOpInterface
  MLIRIR
  )
