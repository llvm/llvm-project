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

add_llvm_example(LLJITWithGDBRegistrationListener
  LLJITWithGDBRegistrationListener.cpp
  )
