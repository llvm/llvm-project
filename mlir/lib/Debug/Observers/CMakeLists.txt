add_mlir_library(MLIRObservers
  ActionLogging.cpp
  ActionProfiler.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Debug/Observers

  LINK_LIBS PUBLIC
  ${LLVM_PTHREAD_LIB}
  MLIRSupport
  MLIRIR
)
