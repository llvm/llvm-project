set(LLVM_LINK_COMPONENTS
  DebugInfoCodeView
  DebugInfoMSF
  DebugInfoPDB
  )

set(DebugInfoPDBSources
  HashTableTest.cpp
  MappedBlockStreamTest.cpp
  NameHashTableBuilderTest.cpp
  MSFBuilderTest.cpp
  PDBApiTest.cpp
  )

add_llvm_unittest(DebugInfoPDBTests
  ${DebugInfoPDBSources}
  )
