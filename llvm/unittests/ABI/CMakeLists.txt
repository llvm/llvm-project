set(LLVM_LINK_COMPONENTS
  ABI
  Support
  TargetParser
  )

add_llvm_unittest(ABITests
  TypesTest.cpp
  X86Test.cpp
  )
