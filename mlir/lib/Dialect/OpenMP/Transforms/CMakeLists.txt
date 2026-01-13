add_mlir_dialect_library(MLIROpenMPTransforms
  OpenMPOffloadPrivatizationPrepare.cpp

  DEPENDS
  MLIROpenMPPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRFuncDialect
  MLIRLLVMDialect
  MLIROpenMPDialect
  MLIRPass
  MLIRTransforms
  )
