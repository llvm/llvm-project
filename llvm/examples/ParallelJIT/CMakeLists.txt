set(LLVM_LINK_COMPONENTS jit interpreter nativecodegen)

add_llvm_example(ParallelJIT
  ParallelJIT.cpp
  )

target_link_libraries(ParallelJIT pthread)
