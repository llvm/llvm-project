add_mlir_dialect_library(MLIRGPUUtils
  Utils.cpp
  DistributionUtils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/GPU/Utils

  # GPU utility headers include llvm/IR/Attributes.h.
  DEPENDS
  intrinsics_gen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRAffineDialect
  MLIRGPUDialect
  MLIRSupport
  MLIRIR
  MLIRVectorDialect
  )
