set(LLVM_LINK_COMPONENTS
  Core
  EmbeddedJIT
  EJIT
  ExecutionEngine
  OrcJIT
  Support
  )

add_llvm_unittest(EJITTests
  EJitRuntimeTest.cpp
  )
