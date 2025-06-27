add_subdirectory(Observers)

add_mlir_library(MLIRDebug
  CLOptionsSetup.cpp
  DebugCounter.cpp
  ExecutionContext.cpp
  BreakpointManagers/FileLineColLocBreakpointManager.cpp
  DebuggerExecutionContextHook.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Debug

  LINK_LIBS PUBLIC
  ${LLVM_PTHREAD_LIB}
  MLIRIR
  MLIRSupport
  MLIRObservers
)

set_target_properties(obj.MLIRDebug PROPERTIES UNITY_BUILD_BATCH_SIZE 1)
