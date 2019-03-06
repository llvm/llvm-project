set(LLVM_LINK_COMPONENTS
  Remarks
  Support
  )

add_llvm_unittest(RemarksTests
  RemarksParsingTest.cpp
  )
