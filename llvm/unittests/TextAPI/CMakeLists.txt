set(LLVM_LINK_COMPONENTS
  TextAPI
)

add_llvm_unittest(TextAPITests
  TextStubV1Tests.cpp
  TextStubV2Tests.cpp
)
