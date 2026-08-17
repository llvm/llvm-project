set(LLVM_LINK_COMPONENTS
  BitReader
  BitWriter
  Core
  LTO
  Support
  )

add_llvm_unittest(LTOTests
  LTOConfigBitcodeTest.cpp
  )

target_link_libraries(LTOTests PRIVATE LLVMTestingSupport)
