# Solaris ld requires this dependency to be made explicit for shared builds.
set(LLVM_LINK_COMPONENTS
  Core
  )

add_polly_unittest(DeLICMTests
  DeLICMTest.cpp
  )
