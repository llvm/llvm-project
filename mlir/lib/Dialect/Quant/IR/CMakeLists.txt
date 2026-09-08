add_mlir_dialect_library(MLIRQuantDialect
  QuantDialectBytecode.h
  QuantDialectBytecode.cpp
  QuantOps.cpp
  QuantTypes.cpp
  TypeDetail.h
  TypeParser.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/QuantOps/IR

  DEPENDS
  MLIRQuantOpsIncGen
  MLIRQuantDialectBytecodeIncGen

  # QuantOps.cpp includes mlir/Transforms/InliningUtils.h.
  HEADER_LIBS
  MLIRTransformUtils

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
  MLIRSupport
  )
