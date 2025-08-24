add_mlir_dialect_library(MLIRTargetLLVMIRTransforms
  TargetToDataLayout.cpp

  DEPENDS
  MLIRTargetLLVMIRTransformsIncGen

  LINK_COMPONENTS
  MC
  Target
  TargetParser
  AllTargetsAsmParsers
  AllTargetsCodeGens
  AllTargetsDescs
  AllTargetsInfos

  LINK_LIBS PUBLIC
  MLIRDLTIDialect
  MLIRLLVMDialect
  MLIRPass
  MLIRTargetLLVMIRImport
  )
