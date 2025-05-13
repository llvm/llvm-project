set(LLVM_LINK_COMPONENTS
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
