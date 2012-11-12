set(LLVM_LINK_COMPONENTS jit interpreter nativecodegen)

add_llvm_example(ParallelJIT
  ParallelJIT.cpp
  )

if(HAVE_LIBPTHREAD)
  target_link_libraries(ParallelJIT pthread)
endif(HAVE_LIBPTHREAD)
