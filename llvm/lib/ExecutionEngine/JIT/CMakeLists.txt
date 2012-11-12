# TODO: Support other architectures. See Makefile.
add_definitions(-DENABLE_X86_JIT)

add_llvm_library(LLVMJIT
  JIT.cpp
  JITDwarfEmitter.cpp
  JITEmitter.cpp
  JITMemoryManager.cpp
  )
