set(LLVM_LINK_COMPONENTS
  ABI
  Core
  Support
  )

add_llvm_unittest(ABITests
  AArch64TargetInfoTest.cpp
  )
