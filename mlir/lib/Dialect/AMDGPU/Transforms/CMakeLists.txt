add_mlir_dialect_library(MLIRAMDGPUTransforms
  EmulateAtomics.cpp
  ResolveStridedMetadata.cpp
  MaskedloadToLoad.cpp

  ADDITIONAL_HEADER_DIRS
  {$MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/AMDGPU/Transforms

  DEPENDS
  MLIRAMDGPUTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRAMDGPUDialect
  MLIRAMDGPUUtils
  MLIRArithDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRVectorDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  )
