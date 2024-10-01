set(LLVM_LINK_COMPONENTS
  Analysis
  Core
  Vectorize
  AsmParser
  TargetParser
  SandboxIR
  )

add_llvm_unittest(SandboxVectorizerTests
  DependencyGraphTest.cpp
  IntervalTest.cpp
  LegalityTest.cpp
  )
