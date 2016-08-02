set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  MC
  MCDisassembler
  Support
  )

add_llvm_unittest(MCTests
  Disassembler.cpp
  StringTableBuilderTest.cpp
  YAMLTest.cpp
  )
