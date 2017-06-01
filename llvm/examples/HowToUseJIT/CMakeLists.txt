set(LLVM_LINK_COMPONENTS
  Core
  ExecutionEngine
  Interpreter
  Support
  nativecodegen
  )

add_llvm_example(HowToUseJIT
  HowToUseJIT.cpp
  )
