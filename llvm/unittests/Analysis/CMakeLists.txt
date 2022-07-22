set(LLVM_LINK_COMPONENTS
  Analysis
  AsmParser
  Core
  Support
  TransformUtils
  )

set(MLGO_TESTS TFUtilsTest.cpp)
if (DEFINED LLVM_HAVE_TF_API)
  LIST(APPEND EXTRA_TESTS ${MLGO_TESTS})
else()
  LIST(APPEND LLVM_OPTIONAL_SOURCES ${MLGO_TESTS})
endif()

add_llvm_unittest_with_input_files(AnalysisTests
  AliasAnalysisTest.cpp
  AliasSetTrackerTest.cpp
  AssumeBundleQueriesTest.cpp
  BasicAliasAnalysisTest.cpp
  BlockFrequencyInfoTest.cpp
  BranchProbabilityInfoTest.cpp
  CallGraphTest.cpp
  CaptureTrackingTest.cpp
  CFGTest.cpp
  CGSCCPassManagerTest.cpp
  ConstraintSystemTest.cpp
  DDGTest.cpp
  DivergenceAnalysisTest.cpp
  DomTreeUpdaterTest.cpp
  GlobalsModRefTest.cpp
  FunctionPropertiesAnalysisTest.cpp
  InlineCostTest.cpp
  IRSimilarityIdentifierTest.cpp
  IVDescriptorsTest.cpp
  LazyCallGraphTest.cpp
  LoadsTest.cpp
  LoopInfoTest.cpp
  LoopNestTest.cpp
  MemoryBuiltinsTest.cpp
  MemoryProfileInfoTest.cpp
  MemorySSATest.cpp
  MLModelRunnerTest.cpp
  PhiValuesTest.cpp
  ProfileSummaryInfoTest.cpp
  ScalarEvolutionTest.cpp
  VectorFunctionABITest.cpp
  SparsePropagation.cpp
  TargetLibraryInfoTest.cpp
  TensorSpecTest.cpp
  TBAATest.cpp
  UnrollAnalyzerTest.cpp
  ValueLatticeTest.cpp
  ValueTrackingTest.cpp
  VectorUtilsTest.cpp
  ${EXTRA_TESTS}
  )

 target_link_libraries(AnalysisTests PRIVATE LLVMTestingSupport)
