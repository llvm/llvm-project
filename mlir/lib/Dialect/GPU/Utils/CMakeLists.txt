add_mlir_dialect_library(MLIRGPUUtils
  Utils.cpp
  DistributionUtils.cpp
  ReductionUtils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/GPU/Utils

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRArithDialect
  MLIRAMDGPUDialect
  MLIRGPUDialect
  MLIRROCDLDialect
  MLIRSupport
  MLIRIR
  )
