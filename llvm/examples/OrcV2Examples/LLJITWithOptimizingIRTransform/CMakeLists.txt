set(LLVM_LINK_COMPONENTS
  Core
  ExecutionEngine
  IPO
  IRReader
  OrcJIT
  OrcShared
  ScalarOpts
  Support
  nativecodegen
  )

add_llvm_example(LLJITWithOptimizingIRTransform
  LLJITWithOptimizingIRTransform.cpp
  )
