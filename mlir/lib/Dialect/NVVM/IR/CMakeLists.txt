add_mlir_dialect_library(MLIRNVVMDialect
  NVVMDialect.cpp
  BasicPtxBuilderInterface.cpp
  NVVMRequiresSMTraits.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/NVVM

  DEPENDS
  MLIRGPUCompilationAttrInterfacesIncGen
  MLIRNVVMOpsIncGen
  MLIRNVVMConversionsIncGen
  MLIRBasicPtxBuilderInterfaceIncGen
  MLIRNVVMRequiresSMTraitsIncGen
  intrinsics_gen

  LINK_COMPONENTS
  AsmParser
  Core
  TargetParser

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
  MLIRGPUDialect
  MLIROpenMPDialect
  MLIRSideEffectInterfaces
  MLIRInferIntRangeInterface
  )
