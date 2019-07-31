set(LLVM_LINK_COMPONENTS
  BitReader
  Remarks
  Support
  )

add_llvm_unittest(RemarksTests
  BitstreamRemarksFormatTest.cpp
  BitstreamRemarksSerializerTest.cpp
  RemarksStrTabParsingTest.cpp
  YAMLRemarksParsingTest.cpp
  YAMLRemarksSerializerTest.cpp
  )
