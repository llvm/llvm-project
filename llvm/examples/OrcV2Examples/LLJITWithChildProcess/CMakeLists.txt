set(LLVM_LINK_COMPONENTS
  Core
  ExecutionEngine
  IRReader
  JITLink
  OrcError
  OrcJIT
  Support
  nativecodegen
  )

add_llvm_example(LLJITInChildProcess
  LLJITWithChildProcess.cpp
  )
