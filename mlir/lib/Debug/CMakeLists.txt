add_mlir_library(MLIRDebug
  DebugCounter.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Debug

  LINK_LIBS PUBLIC
  ${LLVM_PTHREAD_LIB}
  MLIRIR
  MLIRSupport
)
