set(LLVM_LINK_COMPONENTS
  Support
  )

add_llvm_unittest(TestingADTTests
  StringMapTest.cpp
  )
