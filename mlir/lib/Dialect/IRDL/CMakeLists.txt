add_mlir_dialect_library(MLIRIRDL
  IR/IRDL.cpp
  IR/IRDLOps.cpp
  IRDLLoading.cpp
  IRDLVerifiers.cpp

  DEPENDS
  MLIRIRDLIncGen
  MLIRIRDLOpsIncGen
  MLIRIRDLTypesIncGen

  LINK_LIBS PUBLIC
  MLIRDialect
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
  )
