set(LLVM_LINK_COMPONENTS
  BinaryFormat
  )

add_llvm_unittest(BinaryFormatTests
  DwarfTest.cpp
  TestFileMagic.cpp
  )

