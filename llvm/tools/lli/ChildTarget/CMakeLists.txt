set(LLVM_LINK_COMPONENTS
  OrcJIT
  Support
  )

add_llvm_executable(lli-child-target
  ChildTarget.cpp
)

set_target_properties(lli-child-target PROPERTIES FOLDER "Misc")
