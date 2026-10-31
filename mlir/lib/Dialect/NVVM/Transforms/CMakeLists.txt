add_mlir_dialect_library(MLIRNVVMTransforms
  OptimizeForNVVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/NVVM

  DEPENDS
  MLIRNVVMPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
  MLIRNVVMDialect
  MLIRPass
  MLIRTransforms
  )
