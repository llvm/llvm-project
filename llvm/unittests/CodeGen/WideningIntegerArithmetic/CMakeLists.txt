set(LLVM_LINK_COMPONENTS Core Support)

add_llvm_unittest(WideningIntegerArithmeticTests
  WideningIntegerArithmeticTest.cpp
  )
