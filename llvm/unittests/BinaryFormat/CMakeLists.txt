set(LLVM_LINK_COMPONENTS
  BinaryFormat
  )

add_llvm_unittest(BinaryFormatTests
  DwarfTest.cpp
  MachOTest.cpp
  MsgPackReaderTest.cpp
  MsgPackWriterTest.cpp
  TestFileMagic.cpp
  )

