add_mlir_dialect_library(MLIRX86Vector
  X86VectorDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/X86Vector

  DEPENDS
  MLIRX86VectorIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMIR
  MLIRSideEffectInterfaces
  )
