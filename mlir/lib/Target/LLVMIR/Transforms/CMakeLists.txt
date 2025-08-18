add_mlir_dialect_library(MLIRTargetLLVMIRTransforms
  TargetToDataLayout.cpp

  DEPENDS
  MLIRTargetLLVMIRTransformsIncGen

  LINK_COMPONENTS
  #Core
  #IPO
  #IRReader
  #Linker
  MC
  #Passes
  #Support
  Target
  TargetParser
  AllTargetsAsmParsers
  AllTargetsCodeGens
  AllTargetsDescs
  AllTargetsInfos

  LINK_LIBS PUBLIC
  MLIRDLTIDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRPass
  MLIRTargetLLVM
  MLIRTargetLLVMIRImport
  MLIRTransforms
  )
