set(LLVM_LINK_COMPONENTS
  Core
  ExecutionEngine
  IRReader
  JITLink
  OrcJIT
  OrcShared
  OrcTargetProcess
  Support
  nativecodegen
  )

if (LLVM_INCLUDE_UTILS AND NOT LLVM_INCLUDE_UTILS)
  add_llvm_example(LLJITWithRemoteDebugging
    LLJITWithRemoteDebugging.cpp
    RemoteJITUtils.cpp

    DEPENDS
      llvm-jitlink-executor
  )
else()
  # Use a temporary no-op target until D104694 lands.
  add_custom_target(LLJITWithRemoteDebugging)
endif()
