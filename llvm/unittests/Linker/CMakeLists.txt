set(LLVM_LINK_COMPONENTS
  AsmParser
  core
  linker
  )

add_llvm_unittest(LinkerTests
  LinkModulesTest.cpp
  )
