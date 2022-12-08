set(MLGO_TESTS TFUtilsTest.cpp TrainingLoggerTest.cpp)

set(LLVM_OPTIONAL_SOURCES 
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
  InlineAdvisorPlugin.cpp
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
  ${MLGO_TESTS})



# If plugins are disabled, the PluginInlineAdvisor test will disable itself at runtime. 
# Otherwise, reconfiguring with plugins disabled will leave behind a stale executable.
if (LLVM_ENABLE_PLUGINS)
  add_definitions(-DLLVM_ENABLE_PLUGINS)
endif()

# The advisor plugin expects to not link against the Analysis, Support and Core 
# libraries, but expects them to exist in the process loading the plugin. This 
# doesn't work with DLLs on Windows (where a shared library can't have undefined
# references), so just skip this testcase on Windows.
if (NOT WIN32)
  # On AIX, enable run-time linking to allow symbols from the plugins shared
  # objects to be properly bound.
  if(CMAKE_SYSTEM_NAME STREQUAL "AIX")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-brtl")
  endif()


  add_llvm_library(InlineAdvisorPlugin MODULE BUILDTREE_ONLY InlineAdvisorPlugin.cpp)
  # Put PLUGIN next to the unit test executable.
  set_output_directory(InlineAdvisorPlugin
      BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}
      LIBRARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}
      )
  set_target_properties(InlineAdvisorPlugin PROPERTIES FOLDER "Tests")

  set(LLVM_LINK_COMPONENTS
      Analysis
      AsmParser
      Core
      Passes
      Support
  )
  add_llvm_unittest(PluginInlineAdvisorAnalysisTest PluginInlineAdvisorAnalysisTest.cpp)
  target_link_libraries(PluginInlineAdvisorAnalysisTest PRIVATE LLVMTestingSupport)
  export_executable_symbols_for_plugins(PluginInlineAdvisorAnalysisTest)
  add_dependencies(PluginInlineAdvisorAnalysisTest InlineAdvisorPlugin)
  set_property(TARGET PluginInlineAdvisorAnalysisTest PROPERTY FOLDER "Tests/UnitTests/AnalysisTests")
  set_property(TARGET InlineAdvisorPlugin PROPERTY FOLDER "Tests/UnitTests/AnalysisTests")
endif()

if (DEFINED LLVM_HAVE_TF_API)
  LIST(APPEND EXTRA_TESTS ${MLGO_TESTS})
endif()


set(LLVM_LINK_COMPONENTS
  Analysis
  AsmParser
  Core
  Support
  TransformUtils
  )

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
