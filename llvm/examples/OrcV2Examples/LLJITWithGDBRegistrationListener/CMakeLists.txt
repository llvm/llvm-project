set(LLVM_LINK_COMPONENTS
  Core
  ExecutionEngine
  IRReader
  JITLink
  OrcJIT
  OrcTargetProcess
  Support
  nativecodegen
  )

add_llvm_example(LLJITWithGDBRegistrationListener
  LLJITWithGDBRegistrationListener.cpp
  )
