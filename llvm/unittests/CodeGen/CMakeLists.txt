set(LLVM_LINK_COMPONENTS
  AsmPrinter
  Support
  )

set(CodeGenSources
  DIEHashTest.cpp
  )

add_llvm_unittest(CodeGenTests
  ${CodeGenSources}
  )

add_subdirectory(GlobalISel)
