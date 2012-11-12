add_llvm_library(LLVMInstrumentation
  AddressSanitizer.cpp
  BlackList.cpp
  BoundsChecking.cpp
  EdgeProfiling.cpp
  GCOVProfiling.cpp
  Instrumentation.cpp
  OptimalEdgeProfiling.cpp
  PathProfiling.cpp
  ProfilingUtils.cpp
  ThreadSanitizer.cpp
  )

add_dependencies(LLVMInstrumentation intrinsics_gen)
