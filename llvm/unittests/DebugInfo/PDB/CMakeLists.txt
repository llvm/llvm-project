set(LLVM_LINK_COMPONENTS
  DebugInfoCodeView
  DebugInfoMSF
  DebugInfoPDB
  Object
  )

set(LLVM_OPTIONAL_SOURCES
  MSFZReaderTest.cpp
  )

if(LLVM_ENABLE_ZSTD)
  set(msfz_reader_test MSFZReaderTest.cpp)
endif()

add_llvm_unittest_with_input_files(DebugInfoPDBTests
  HashTableTest.cpp
  NativeSessionTest.cpp
  NativeSymbolReuseTest.cpp
  StringTableBuilderTest.cpp
  PDBApiTest.cpp
  PDBVariantTest.cpp
  PublicsStreamTest.cpp
  TpiStreamTest.cpp
  ${msfz_reader_test}
  )

target_link_libraries(DebugInfoPDBTests PRIVATE LLVMTestingSupport)
