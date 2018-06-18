set(LLVM_LINK_COMPONENTS
  Vectorize
  )

add_llvm_unittest(VectorizeTests
  VPlanTest.cpp
  )
