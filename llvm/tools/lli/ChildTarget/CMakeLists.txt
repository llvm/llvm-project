set(LLVM_LINK_COMPONENTS
  OrcJIT
  RuntimeDyld
  Support
  )

add_llvm_utility(lli-child-target
  ChildTarget.cpp
)

