set(LLVM_LINK_COMPONENTS
  BitReader
  Remarks
  Support
  )

add_llvm_unittest(RemarksTests
  BitstreamRemarksFormatTest.cpp
  BitstreamRemarksParsingTest.cpp
  BitstreamRemarksSerializerTest.cpp
  RemarksAPITest.cpp
  RemarksStrTabParsingTest.cpp
  YAMLRemarksParsingTest.cpp
  YAMLRemarksSerializerTest.cpp
  )
