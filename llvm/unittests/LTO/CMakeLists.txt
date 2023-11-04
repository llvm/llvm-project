set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  LTO
  )

add_llvm_unittest(LTOTests
  LTOTest.cpp
  )
