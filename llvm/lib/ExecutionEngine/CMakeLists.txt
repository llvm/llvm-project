

add_llvm_library(LLVMExecutionEngine
  ExecutionEngine.cpp
  ExecutionEngineBindings.cpp
  TargetSelect.cpp
  )

add_subdirectory(Interpreter)
add_subdirectory(JIT)
add_subdirectory(MCJIT)
add_subdirectory(RuntimeDyld)

if( LLVM_USE_OPROFILE )
  add_subdirectory(OProfileJIT)
endif( LLVM_USE_OPROFILE )

if( LLVM_USE_INTEL_JITEVENTS )
  add_subdirectory(IntelJITEvents)
endif( LLVM_USE_INTEL_JITEVENTS )
