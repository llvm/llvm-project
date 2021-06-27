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

if ( LLVM_INCLUDE_UTILS )
  add_llvm_example(LLJITWithRemoteDebugging
    LLJITWithRemoteDebugging.cpp
    RemoteJITUtils.cpp

    DEPENDS
      llvm-jitlink-executor
  )
endif()
