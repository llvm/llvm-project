set(LLVM_LINK_COMPONENTS
  Analysis
  Core
  Vectorize
  AsmParser
  TargetParser
  )

add_llvm_unittest(VectorizeTests
  VPlanTest.cpp
  VPDomTreeTest.cpp
  VPlanHCFGTest.cpp
  VPlanSlpTest.cpp
  VPlanVerifierTest.cpp
  )
