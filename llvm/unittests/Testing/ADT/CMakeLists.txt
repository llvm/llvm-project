set(LLVM_LINK_COMPONENTS
  Support
  )

add_llvm_unittest(TestingADTTests
  StringMapEntryTest.cpp
  StringMapTest.cpp
  )
