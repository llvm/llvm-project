set(LLVM_LINK_COMPONENTS
  ABI
  Core
  Support
  )

add_llvm_unittest(ABITests
  AArch64TargetInfoTest.cpp
  FunctionInfoTest.cpp
  IRTypeMapperTest.cpp
  X86TargetInfoTest.cpp
  TypesTest.cpp
  )
