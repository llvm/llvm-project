set(LLVM_LINK_COMPONENTS
  AsmParser
  core
  linker
  )

set(LinkerSources
  LinkModulesTest.cpp
  )

add_llvm_unittest(LinkerTests
  ${LinkerSources}
  )
