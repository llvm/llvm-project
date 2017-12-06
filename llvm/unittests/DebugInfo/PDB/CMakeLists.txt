set(LLVM_LINK_COMPONENTS
  DebugInfoCodeView
  DebugInfoMSF
  DebugInfoPDB
  )

set(DebugInfoPDBSources
  HashTableTest.cpp
  StringTableBuilderTest.cpp
  PDBApiTest.cpp
  )

add_llvm_unittest(DebugInfoPDBTests
  ${DebugInfoPDBSources}
  )

target_link_libraries(DebugInfoPDBTests PRIVATE LLVMTestingSupport)
