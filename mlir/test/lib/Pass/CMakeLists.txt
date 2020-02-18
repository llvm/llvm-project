add_llvm_library(MLIRTestPass
  TestPassManager.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Pass

  LINK_LIBS
  MLIRIR
  MLIRPass
  LLVMSupport
  )
