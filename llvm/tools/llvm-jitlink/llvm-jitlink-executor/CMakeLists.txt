set(LLVM_LINK_COMPONENTS
  ExecutionEngine
  OrcShared
  OrcTargetProcess
  Support
  )

add_llvm_utility(llvm-jitlink-executor
  llvm-jitlink-executor.cpp

  DEPENDS
  intrinsics_gen

  EXPORT_SYMBOLS
)
