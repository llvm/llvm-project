set(LLVM_TARGET_DEFINITIONS ArithmeticCanonicalization.td)
mlir_tablegen(ArithmeticCanonicalization.inc -gen-rewriters)
add_public_tablegen_target(MLIRArithmeticCanonicalizationIncGen)

add_mlir_dialect_library(MLIRArithmeticDialect
  ArithmeticOps.cpp
  ArithmeticDialect.cpp
  InferIntRangeInterfaceImpls.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Arithmetic

  DEPENDS
  MLIRArithmeticOpsIncGen

  LINK_LIBS PUBLIC
  MLIRDialect
  MLIRDialectUtils
  MLIRInferIntRangeInterface
  MLIRInferTypeOpInterface
  MLIRIR
  )
