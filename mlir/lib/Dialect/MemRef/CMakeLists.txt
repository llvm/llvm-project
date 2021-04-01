add_mlir_dialect_library(MLIRMemRef
  IR/MemRefDialect.cpp
  IR/MemRefOps.cpp
  Utils/MemRefUtils.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/inlude/mlir/Dialect/MemRefDialect

  DEPENDS
  MLIRStandardOpsIncGen
  MLIRMemRefOpsIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRDialect
  MLIRInferTypeOpInterface
  MLIRIR
  MLIRStandard
  MLIRTensor
  MLIRViewLikeInterface
)
