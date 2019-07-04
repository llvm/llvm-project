set(LLVM_LINK_COMPONENTS
  BitstreamReader
  )

add_llvm_unittest(BitstreamTests
  BitstreamReaderTest.cpp
  BitstreamWriterTest.cpp
  )
