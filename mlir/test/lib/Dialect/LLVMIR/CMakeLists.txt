
add_mlir_library(MLIRLLVMTypeTestDialect
  LLVMTypeTestDialect.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRDialect
  MLIRIR
  MLIRLLVMIR
  )
