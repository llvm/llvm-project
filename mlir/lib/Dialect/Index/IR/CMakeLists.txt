add_mlir_dialect_library(MLIRIndexDialect
  IndexAttrs.cpp
  IndexDialect.cpp
  IndexOps.cpp

  DEPENDS
  MLIRIndexOpsIncGen

  LINK_LIBS PUBLIC
  MLIRDialect
  MLIRIR
  MLIRCastInterfaces
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
  )
