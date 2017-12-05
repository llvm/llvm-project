set(LLVM_LINK_COMPONENTS
  Core
  ExecutionEngine
  Interpreter
  MC
  Support
  nativecodegen
  )

add_llvm_example(ParallelJIT
  ParallelJIT.cpp
  )

target_link_libraries(ParallelJIT PRIVATE ${LLVM_PTHREAD_LIB})
