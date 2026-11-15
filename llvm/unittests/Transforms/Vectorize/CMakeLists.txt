add_subdirectory(SandboxVectorizer)

# VPlanVerifierTest.cpp includes the generated VectorizeOptions.inc.
include_directories(${LLVM_BINARY_DIR}/lib/Transforms/Vectorize)

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
  VPPostDomFrontierTest.cpp
  VPlanHCFGTest.cpp
  VPlanPatternMatchTest.cpp
  VPlanVerifierTest.cpp
  )
