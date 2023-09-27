# Solaris ld requires this dependency to be made explicit for shared builds.
set(LLVM_LINK_COMPONENTS
  Core
  SelectionDAG
  )

add_polly_unittest(DeLICMTests
  DeLICMTest.cpp
  )
