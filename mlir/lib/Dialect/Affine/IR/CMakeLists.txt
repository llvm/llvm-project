add_mlir_dialect_library(MLIRAffine
  AffineMemoryOpInterfaces.cpp
  AffineOps.cpp
  AffineValueMap.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Affine

  DEPENDS
  MLIRAffineMemoryOpInterfacesIncGen
  MLIRAffineOpsIncGen

  LINK_LIBS PUBLIC
  MLIREDSC
  MLIRIR
  MLIRLoopLikeInterface
  MLIRMemRef
  MLIRSideEffectInterfaces
  MLIRStandard
  )
