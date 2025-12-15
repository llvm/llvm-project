set(LLVM_LINK_COMPONENTS
  Core
  IRReader
  OrcJIT
  OrcTargetProcess
  ExecutionEngine
  Support
  nativecodegen
  Analysis
  Passes
  )

if(LLVM_ENABLE_THREADS)
  add_llvm_example(SpeculativeJIT
    SpeculativeJIT.cpp
  )
endif()
