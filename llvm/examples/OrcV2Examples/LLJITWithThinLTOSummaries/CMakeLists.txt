set(LLVM_LINK_COMPONENTS
  BitReader
  Core
  ExecutionEngine
  OrcJIT
  OrcTargetProcess
  Support
  nativecodegen
  )

add_llvm_example(LLJITWithThinLTOSummaries
  LLJITWithThinLTOSummaries.cpp
  )
