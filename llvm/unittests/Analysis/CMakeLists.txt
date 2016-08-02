set(LLVM_LINK_COMPONENTS
  Analysis
  AsmParser
  Core
  Support
  )

add_llvm_unittest(AnalysisTests
  AliasAnalysisTest.cpp
  CallGraphTest.cpp
  CFGTest.cpp
  LazyCallGraphTest.cpp
  ScalarEvolutionTest.cpp
  MixedTBAATest.cpp
  ValueTrackingTest.cpp
  )
