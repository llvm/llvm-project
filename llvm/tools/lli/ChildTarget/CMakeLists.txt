set(LLVM_LINK_COMPONENTS
  OrcError
  OrcJIT
  RuntimeDyld
  Support
  )

add_llvm_utility(lli-child-target
  ChildTarget.cpp

  DEPENDS
  intrinsics_gen
)

