add_llvm_library(LLVMVectorize
  BBVectorize.cpp
  Vectorize.cpp
  LoopVectorize.cpp
  )

add_dependencies(LLVMVectorize intrinsics_gen)
