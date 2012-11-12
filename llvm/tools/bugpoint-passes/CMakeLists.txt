add_llvm_loadable_module( BugpointPasses
  TestPasses.cpp
  )

add_dependencies(BugpointPasses bugpoint)
