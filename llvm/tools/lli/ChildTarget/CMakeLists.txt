set(LLVM_LINK_COMPONENTS
  OrcShared
  OrcJIT
  RuntimeDyld
  Support
  )

add_llvm_utility(lli-child-target
  ChildTarget.cpp

  DEPENDS
  intrinsics_gen
)

