set(LLVM_LINK_COMPONENTS
  BinaryFormat
  TargetParser
  )

add_llvm_unittest(BinaryFormatTests
  DwarfTest.cpp
  ELFTest.cpp
  MachOTest.cpp
  MsgPackDocumentTest.cpp
  MsgPackReaderTest.cpp
  MsgPackWriterTest.cpp
  TestFileMagic.cpp
  )

