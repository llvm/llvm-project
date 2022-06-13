add_mlir_dialect_library(MLIRAffineDialect
  AffineMemoryOpInterfaces.cpp
  AffineOps.cpp
  AffineValueMap.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Affine

  DEPENDS
  MLIRAffineMemoryOpInterfacesIncGen
  MLIRAffineOpsIncGen

  LINK_LIBS PUBLIC
  MLIRArithmeticDialect
  MLIRIR
  MLIRLoopLikeInterface
  MLIRMemRefDialect
  MLIRSideEffectInterfaces
  MLIRTensorDialect
  )
