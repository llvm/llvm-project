set(LLVM_LINK_COMPONENTS
  Core
  ExecutionEngine
  IRReader
  OrcJIT
  OrcShared
  Support
  nativecodegen
  )

add_llvm_example(LLJITWithObjectCache
  LLJITWithObjectCache.cpp
  )
