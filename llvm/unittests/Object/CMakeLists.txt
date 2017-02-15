set(LLVM_LINK_COMPONENTS
  Object
  )

add_llvm_unittest(ObjectTests
  SymbolSizeTest.cpp
  SymbolicFileTest.cpp
  )

