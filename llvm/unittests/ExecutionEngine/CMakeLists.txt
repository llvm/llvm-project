set(LLVM_LINK_COMPONENTS
  interpreter
  )

add_llvm_unittest(ExecutionEngineTests
  ExecutionEngineTest.cpp
  )

add_subdirectory(JIT)
add_subdirectory(MCJIT)
