set(LLVM_LINK_COMPONENTS
  OptRemarks
  Support
  )

add_llvm_unittest(OptRemarksTests
  OptRemarksParsingTest.cpp
  )
