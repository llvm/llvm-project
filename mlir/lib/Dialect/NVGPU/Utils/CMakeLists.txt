add_mlir_dialect_library(MLIRNVGPUUtils
  MMAUtils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Arithmetic/Utils

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRLLVMDialect
  MLIRNVGPUDialect
  MLIRNVVMDialect
  MLIRVectorDialect
  MLIRIR
  )
