set(LLVM_LINK_COMPONENTS
  DebugInfoCodeView
  DebugInfoMSF
  DebugInfoPDB
  TestingSupport
  )

set(DebugInfoPDBSources
  HashTableTest.cpp
  MappedBlockStreamTest.cpp
  StringTableBuilderTest.cpp
  MSFBuilderTest.cpp
  PDBApiTest.cpp
  TypeServerHandlerTest.cpp
  )

add_llvm_unittest(DebugInfoPDBTests
  ${DebugInfoPDBSources}
  )
