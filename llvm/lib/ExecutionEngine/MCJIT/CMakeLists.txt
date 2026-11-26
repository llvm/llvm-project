add_llvm_component_library(LLVMMCJIT
  MCJIT.cpp

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  Core
  ExecutionEngine
  MC
  Object
  RuntimeDyld
  Support
  Target
  TargetParser
  )
