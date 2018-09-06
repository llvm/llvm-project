set(LLVM_LINK_COMPONENTS
  DebugInfoCodeView
  DebugInfoMSF
  DebugInfoPDB
  )

add_llvm_unittest(DebugInfoPDBTests
  HashTableTest.cpp
  NativeSymbolReuseTest.cpp
  StringTableBuilderTest.cpp
  PDBApiTest.cpp
  )

target_link_libraries(DebugInfoPDBTests PRIVATE LLVMTestingSupport)
