set(LLVM_LINK_COMPONENTS jit interpreter nativecodegen)

add_llvm_example(Fibonacci
  fibonacci.cpp
  )
