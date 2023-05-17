set(LLVM_LINK_COMPONENTS
  BinaryFormat
  Object
  ObjectYAML
  TargetParser
  )

add_llvm_unittest(ObjectTests
  ArchiveTest.cpp
  DXContainerTest.cpp
  ELFObjectFileTest.cpp
  ELFTypesTest.cpp
  ELFTest.cpp
  GOFFObjectFileTest.cpp
  MinidumpTest.cpp
  ObjectFileTest.cpp
  OffloadingTest.cpp
  SymbolSizeTest.cpp
  SymbolicFileTest.cpp
  XCOFFObjectFileTest.cpp
  )

target_link_libraries(ObjectTests PRIVATE LLVMTestingSupport)

if (MSVC)
  # Disable warning C4309: '=': truncation of constant value
  set_source_files_properties(GOFFObjectFileTest.cpp PROPERTIES COMPILE_FLAGS -wd4309)
endif()
