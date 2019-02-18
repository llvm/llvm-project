set(LLVM_LINK_COMPONENTS
  Analysis
  Core
  Vectorize
  AsmParser
  )

add_llvm_unittest(VectorizeTests
  VPlanDominatorTreeTest.cpp
  VPlanLoopInfoTest.cpp
  VPlanTest.cpp
  VPlanHCFGTest.cpp
  VPlanSlpTest.cpp
  )
