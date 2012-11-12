set(LLVM_LINK_COMPONENTS
  Analysis
  )

add_llvm_unittest(AnalysisTests
  ScalarEvolutionTest.cpp
  )
