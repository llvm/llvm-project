set(LLVM_LINK_COMPONENTS
  ABI
  Support
  )

add_llvm_unittest(ABITests
  TypesTest.cpp
  )
