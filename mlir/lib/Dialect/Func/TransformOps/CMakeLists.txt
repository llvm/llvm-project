add_mlir_dialect_library(MLIRFuncTransformOps
  FuncTransformOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Func/TransformOps

  DEPENDS
  MLIRFuncTransformOpsIncGen

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRFuncToLLVM
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRTransformDialect
)
