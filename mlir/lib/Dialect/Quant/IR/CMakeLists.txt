add_mlir_dialect_library(MLIRQuant
  QuantOps.cpp
  QuantTypes.cpp
  TypeDetail.h
  TypeParser.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/QuantOps/IR

  DEPENDS
  MLIRQuantOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSideEffectInterfaces
  MLIRSupport
  )
