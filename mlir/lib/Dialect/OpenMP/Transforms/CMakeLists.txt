add_mlir_dialect_library(MLIROpenMPTransforms
  MarkDeclareTarget.cpp
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
