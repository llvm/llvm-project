# Build all these tests with -O0, otherwise optimizations may merge some
# basic blocks and we'll fail to discover the targets.
# We change the flags for every build type because we might be doing
# a multi-configuration build (e.g. Xcode) where CMAKE_BUILD_TYPE doesn't
# mean anything.
set(variables_to_filter
  CMAKE_CXX_FLAGS_RELEASE
  CMAKE_CXX_FLAGS_DEBUG
  CMAKE_CXX_FLAGS_RELWITHDEBINFO
  CMAKE_CXX_FLAGS_MINSIZEREL
  LIBFUZZER_FLAGS_BASE
  )
foreach (VARNAME ${variables_to_filter})
  string(REGEX REPLACE "([-/]O)[123s]" "\\10" ${VARNAME} "${${VARNAME}}")
endforeach()

# Enable the coverage instrumentation (it is disabled for the Fuzzer lib).
set(CMAKE_CXX_FLAGS "${LIBFUZZER_FLAGS_BASE} -fsanitize-coverage=trace-pc-guard,indirect-calls,trace-cmp,trace-div,trace-gep -gline-tables-only")

if(MSVC)
  # For tests use the CRT specified for release build
  # (asan doesn't support MDd and MTd)
  if ("${LLVM_USE_CRT_RELEASE}" STREQUAL "")
    set(CRT_FLAG " /MD ")
  else()
    set(CRT_FLAG " /${LLVM_USE_CRT_RELEASE} ")
  endif()
  # In order to use the sanitizers in Windows, we need to link against many
  # runtime libraries which will depend on the target being created
  # (executable or dll) and the c runtime library used (MT/MD).
  # By default, cmake uses link.exe for linking, which fails because we don't
  # specify the appropiate dependencies.
  # As we don't want to consider all of that possible situations which depends
  # on the implementation of the compiler-rt, the simplest option is to change
  # the rules for linking executables and shared libraries, using the compiler
  # instead of link.exe. Clang will consider the sanitizer flags, and
  # automatically provide the required libraries to the linker.
  set(CMAKE_CXX_LINK_EXECUTABLE "<CMAKE_CXX_COMPILER> <FLAGS> ${CMAKE_CXX_FLAGS} ${CRT_FLAG} <OBJECTS> -o <TARGET> <LINK_LIBRARIES> /link <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS>")
  set(CMAKE_CXX_CREATE_SHARED_LIBRARY "<CMAKE_CXX_COMPILER> ${CMAKE_CXX_FLAGS} ${CRT_FLAG} /LD <CMAKE_SHARED_LIBRARY_CXX_FLAGS> <CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS> <CMAKE_SHARED_LIBRARY_SONAME_CXX_FLAG> <TARGET_SONAME> -o <TARGET> <OBJECTS> <LINK_LIBRARIES> /link <LINK_FLAGS>")
endif()

add_custom_target(TestBinaries)

# add_libfuzzer_test(<name>
#   SOURCES source0.cpp [source1.cpp ...]
#   )
#
#   Declares a LibFuzzer test executable with target name LLVMFuzzer-<name>.
#
#   One or more source files to be compiled into the binary must be declared
#   after the SOURCES keyword.
function(add_libfuzzer_test name)
  set(multi_arg_options "SOURCES")
  cmake_parse_arguments(
    "add_libfuzzer_test" "" "" "${multi_arg_options}" ${ARGN})
  if ("${add_libfuzzer_test_SOURCES}" STREQUAL "")
    message(FATAL_ERROR "Source files must be specified")
  endif()
  add_executable(LLVMFuzzer-${name}
    ${add_libfuzzer_test_SOURCES}
    )
  target_link_libraries(LLVMFuzzer-${name} LLVMFuzzer)
  # Place binary where llvm-lit expects to find it
  set_target_properties(LLVMFuzzer-${name}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY
    "${CMAKE_BINARY_DIR}/lib/Fuzzer/test"
    )
  add_dependencies(TestBinaries LLVMFuzzer-${name})
endfunction()

###############################################################################
# Basic tests
###############################################################################

set(Tests
  AbsNegAndConstantTest
  AbsNegAndConstant64Test
  AccumulateAllocationsTest
  BadStrcmpTest
  BogusInitializeTest
  BufferOverflowOnInput
  CallerCalleeTest
  CounterTest
  CustomCrossOverAndMutateTest
  CustomCrossOverTest
  CustomMutatorTest
  CxxStringEqTest
  DivTest
  EmptyTest
  EquivalenceATest
  EquivalenceBTest
  FourIndependentBranchesTest
  FullCoverageSetTest
  InitializeTest
  Memcmp64BytesTest
  MemcmpTest
  LeakTest
  LeakTimeoutTest
  LoadTest
  NullDerefTest
  NullDerefOnEmptyTest
  NthRunCrashTest
  OneHugeAllocTest
  OutOfMemoryTest
  OutOfMemorySingleLargeMallocTest
  RepeatedMemcmp
  RepeatedBytesTest
  SimpleCmpTest
  SimpleDictionaryTest
  SimpleHashTest
  SimpleTest
  SimpleThreadedTest
  SingleByteInputTest
  SingleMemcmpTest
  SingleStrcmpTest
  SingleStrncmpTest
  SpamyTest
  ShrinkControlFlowTest
  ShrinkValueProfileTest
  StrcmpTest
  StrncmpOOBTest
  StrncmpTest
  StrstrTest
  SwapCmpTest
  SwitchTest
  Switch2Test
  TableLookupTest
  ThreadedLeakTest
  ThreadedTest
  TimeoutTest
  TimeoutEmptyTest
  TraceMallocTest
  TwoDifferentBugsTest
  )

if(APPLE OR MSVC)
  # LeakSanitizer is not supported on OSX and Windows right now
  set(HAS_LSAN 0)
  message(WARNING "LeakSanitizer is not supported."
    " Building and running LibFuzzer LeakSanitizer tests is disabled."
    )
else()
  set(HAS_LSAN 1)
endif()

foreach(Test ${Tests})
  add_libfuzzer_test(${Test} SOURCES ${Test}.cpp)
endforeach()

function(test_export_symbol target symbol)
  if(MSVC)
    set_target_properties(LLVMFuzzer-${target} PROPERTIES LINK_FLAGS
        "-export:${symbol}")
  endif()
endfunction()

test_export_symbol(InitializeTest "LLVMFuzzerInitialize")
test_export_symbol(BogusInitializeTest "LLVMFuzzerInitialize")
test_export_symbol(CustomCrossOverTest "LLVMFuzzerCustomCrossOver")
test_export_symbol(CustomMutatorTest "LLVMFuzzerCustomMutator")

###############################################################################
# Unit tests
###############################################################################

add_executable(LLVMFuzzer-Unittest
  FuzzerUnittest.cpp
  )

add_executable(LLVMFuzzer-StandaloneInitializeTest
  InitializeTest.cpp
  ../standalone/StandaloneFuzzTargetMain.c
  )

target_link_libraries(LLVMFuzzer-Unittest
  gtest
  gtest_main
  LLVMFuzzerNoMain
  )

target_include_directories(LLVMFuzzer-Unittest PRIVATE
  "${LLVM_MAIN_SRC_DIR}/utils/unittest/googletest/include"
  )

add_dependencies(TestBinaries LLVMFuzzer-Unittest)
set_target_properties(LLVMFuzzer-Unittest
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY
  "${CMAKE_CURRENT_BINARY_DIR}"
)

add_dependencies(TestBinaries LLVMFuzzer-StandaloneInitializeTest)
set_target_properties(LLVMFuzzer-StandaloneInitializeTest
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY
  "${CMAKE_CURRENT_BINARY_DIR}"
)

###############################################################################
# Additional tests
###############################################################################

include_directories(..)

# add_subdirectory(uninstrumented)
add_subdirectory(no-coverage)
add_subdirectory(trace-pc)
add_subdirectory(ubsan)

add_library(LLVMFuzzer-DSO1 SHARED DSO1.cpp)
add_library(LLVMFuzzer-DSO2 SHARED DSO2.cpp)

add_executable(LLVMFuzzer-DSOTest
  DSOTestMain.cpp
  DSOTestExtra.cpp)

target_link_libraries(LLVMFuzzer-DSOTest
  LLVMFuzzer-DSO1
  LLVMFuzzer-DSO2
  LLVMFuzzer
  )

set_target_properties(LLVMFuzzer-DSOTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY
  "${CMAKE_BINARY_DIR}/lib/Fuzzer/test")

if(MSVC)
  set_output_directory(LLVMFuzzer-DSO1
    BINARY_DIR "${CMAKE_BINARY_DIR}/lib/Fuzzer/test"
    LIBRARY_DIR "${CMAKE_BINARY_DIR}/lib/Fuzzer/test")
  set_output_directory(LLVMFuzzer-DSO2
    BINARY_DIR "${CMAKE_BINARY_DIR}/lib/Fuzzer/test"
    LIBRARY_DIR "${CMAKE_BINARY_DIR}/lib/Fuzzer/test")
else(MSVC)
  set_output_directory(LLVMFuzzer-DSO1
    LIBRARY_DIR "${CMAKE_BINARY_DIR}/lib/Fuzzer/lib")
  set_output_directory(LLVMFuzzer-DSO2
    LIBRARY_DIR "${CMAKE_BINARY_DIR}/lib/Fuzzer/lib")
endif()

add_dependencies(TestBinaries LLVMFuzzer-DSOTest)

###############################################################################
# Configure lit to run the tests
#
# Note this is done after declaring all tests so we can inform lit if any tests
# need to be disabled.
###############################################################################
set(LIBFUZZER_POSIX 1)
if (MSVC)
  set(LIBFUZZER_POSIX 0)
endif()

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
  )

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/unit/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/unit/lit.site.cfg
  )

add_lit_testsuite(check-fuzzer "Running Fuzzer tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS TestBinaries
    )

# Don't add dependencies on Windows. The linker step would fail on Windows,
# since cmake will use link.exe for linking and won't include compiler-rt libs.
if(NOT MSVC)
  add_dependencies(check-fuzzer FileCheck sancov not)
endif()
