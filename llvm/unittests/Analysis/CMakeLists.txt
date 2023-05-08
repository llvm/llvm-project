set(LLVM_LINK_COMPONENTS
  Analysis
  AsmParser
  Core
  Passes
  Support
  TargetParser
  TransformUtils
  IPO
  )

set(ANALYSIS_TEST_SOURCES
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
  PluginInlineAdvisorAnalysisTest.cpp
  PluginInlineOrderAnalysisTest.cpp
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
  )

set(MLGO_TESTS TFUtilsTest.cpp)

if (LLVM_HAVE_TFLITE)
  LIST(APPEND ANALYSIS_TEST_SOURCES ${MLGO_TESTS})
else()
  LIST(APPEND LLVM_OPTIONAL_SOURCES ${MLGO_TESTS})
endif()

add_llvm_unittest_with_input_files(AnalysisTests
  ${ANALYSIS_TEST_SOURCES}
  )

target_link_libraries(AnalysisTests PRIVATE LLVMTestingSupport)

# On AIX, enable run-time linking to allow symbols from the plugins shared
# objects to be properly bound.
if(CMAKE_SYSTEM_NAME STREQUAL "AIX")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-brtl")
endif()

# Export symbols from the plugins shared objects.
if(NOT WIN32)
  export_executable_symbols_for_plugins(AnalysisTests)
endif()

add_subdirectory(InlineAdvisorPlugin)
add_subdirectory(InlineOrderPlugin)
