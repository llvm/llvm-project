set(LLVM_LINK_COMPONENTS
  Support
  )

add_llvm_unittest(MITests
  LiveIntervalTest.cpp
  )
