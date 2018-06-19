set(LLVM_LINK_COMPONENTS
  Vectorize
  AsmParser
  )

add_llvm_unittest(VectorizeTests
  VPlanTest.cpp
  VPlanHCFGTest.cpp
  )
