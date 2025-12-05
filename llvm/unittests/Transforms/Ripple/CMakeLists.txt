set(LLVM_LINK_COMPONENTS
  Ripple
  )

add_llvm_unittest(RippleTests
  TensorShapeTests.cpp
  )
