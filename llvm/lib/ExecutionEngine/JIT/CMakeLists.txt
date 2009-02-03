# TODO: Support other architectures. See Makefile.
add_definitions(-DENABLE_X86_JIT)

add_partially_linked_object(LLVMJIT
  Intercept.cpp
  JIT.cpp
  JITDwarfEmitter.cpp
  JITEmitter.cpp
  JITMemoryManager.cpp
  TargetSelect.cpp
  )
