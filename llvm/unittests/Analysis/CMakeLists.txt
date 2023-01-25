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
  )

# The unit tests target does not use InlineAdvisorPlugin.cpp
# so if not added to LLVM_OPTIONAL_SOURCES, FileCheck will
# complain about unused file.
set(LLVM_OPTIONAL_SOURCES InlineAdvisorPlugin.cpp)

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

  # This plugin is built as a stand-alone plugin, so since we don't use the
  # ANALYSIS_TEST_SOURCES files, we have to add them to LLVM_OPTIONAL_SOURCES
  # so that FileCheck doesn't complain about unsued files.
  LIST(APPEND LLVM_OPTIONAL_SOURCES ${ANALYSIS_TEST_SOURCES})

  unset(LLVM_LINK_COMPONENTS)
  add_llvm_library(InlineAdvisorPlugin MODULE BUILDTREE_ONLY InlineAdvisorPlugin.cpp)
  # Put PLUGIN next to the unit test executable.
  set_output_directory(InlineAdvisorPlugin
      BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}
      LIBRARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}
      )
  set_target_properties(InlineAdvisorPlugin PROPERTIES FOLDER "Tests")

  export_executable_symbols_for_plugins(AnalysisTests)
  # The plugin depends on some of the output files of intrinsics_gen, so make sure
  # it is built before the plugin.
  add_dependencies(InlineAdvisorPlugin intrinsics_gen)
  add_dependencies(AnalysisTests InlineAdvisorPlugin)
  set_property(TARGET InlineAdvisorPlugin PROPERTY FOLDER "Tests/UnitTests/AnalysisTests")
endif()
