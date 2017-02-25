set(LLVM_LINK_COMPONENTS
  DebugInfoCodeView
  DebugInfoMSF
  DebugInfoPDB
  )

set(DebugInfoPDBSources
  BinaryStreamTest.cpp
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
