set(LLVM_LINK_COMPONENTS
  Core
  ExecutionEngine
  IRReader
  OrcJIT
  Support
  nativecodegen
  )

add_llvm_example(LLJITWithObjectCache
  LLJITWithObjectCache.cpp
  )
