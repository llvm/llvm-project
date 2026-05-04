set(LLVM_LINK_COMPONENTS
  Analysis
  Core
  EmbeddedJIT
  EJIT
  ExecutionEngine
  OrcJIT
  Passes
  Support
  TargetParser
  )

add_llvm_unittest(EJITTests
  EJitRuntimeTest.cpp
  )
