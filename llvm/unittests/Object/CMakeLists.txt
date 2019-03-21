set(LLVM_LINK_COMPONENTS
  BinaryFormat
  Object
  )

add_llvm_unittest(ObjectTests
  MinidumpTest.cpp
  SymbolSizeTest.cpp
  SymbolicFileTest.cpp
  )

target_link_libraries(ObjectTests PRIVATE LLVMTestingSupport)
