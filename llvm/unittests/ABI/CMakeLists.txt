set(LLVM_LINK_COMPONENTS
  ABI
  Core
  Support
  )

add_llvm_unittest(ABITests
  AArch64TargetInfoTest.cpp
  SystemZTargetInfoTest.cpp
  IRTypeMapperTest.cpp
  FunctionInfoTest.cpp
  TargetInfoTest.cpp
  AMDGPUTargetInfoTest.cpp
  SPIRVTargetInfoTest.cpp
  X86TargetInfoTest.cpp
  TypesTest.cpp
  )
