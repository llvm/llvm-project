set(LLVM_LINK_COMPONENTS
  BinaryFormat
  Object
  )

add_llvm_unittest(ObjectTests
  ArchiveTest.cpp
  ELFObjectFileTest.cpp
  ELFTypesTest.cpp
  ELFTest.cpp
  MinidumpTest.cpp
  ObjectFileTest.cpp
  SymbolSizeTest.cpp
  SymbolicFileTest.cpp
  XCOFFObjectFileTest.cpp
  )

target_link_libraries(ObjectTests PRIVATE LLVMTestingSupport)
