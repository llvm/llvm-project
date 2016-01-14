# Build all these tests with -O0, otherwise optimizations may merge some
# basic blocks and we'll fail to discover the targets.
# Also enable the coverage instrumentation back (it is disabled
# for the Fuzzer lib)
set(CMAKE_CXX_FLAGS_RELEASE "${LIBFUZZER_FLAGS_BASE} -O0 -fsanitize-coverage=edge,indirect-calls")

set(DFSanTests
  MemcmpTest
  SimpleCmpTest
  StrcmpTest
  StrncmpTest
  SwitchTest
  )

set(Tests
  BufferOverflowOnInput
  CallerCalleeTest
  CounterTest
  FourIndependentBranchesTest
  FullCoverageSetTest
  MemcmpTest
  NullDerefTest
  SimpleCmpTest
  SimpleDictionaryTest
  SimpleHashTest
  SimpleTest
  StrcmpTest
  StrncmpTest
  SwitchTest
  ThreadedTest
  TimeoutTest
  )

set(CustomMainTests
  UserSuppliedFuzzerTest
  )

set(UninstrumentedTests
  UninstrumentedTest
  )

set(TraceBBTests
  SimpleTest
  )

set(TestBinaries)

foreach(Test ${Tests})
  add_executable(LLVMFuzzer-${Test}
    ${Test}.cpp
    )
  target_link_libraries(LLVMFuzzer-${Test}
    LLVMFuzzer
    )
  set(TestBinaries ${TestBinaries} LLVMFuzzer-${Test})
endforeach()

foreach(Test ${CustomMainTests})
  add_executable(LLVMFuzzer-${Test}
    ${Test}.cpp
    )
  target_link_libraries(LLVMFuzzer-${Test}
    LLVMFuzzerNoMain
    )
  set(TestBinaries ${TestBinaries} LLVMFuzzer-${Test})
endforeach()


configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
  )

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/unit/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/unit/lit.site.cfg
  )

include_directories(..)
include_directories(${LLVM_MAIN_SRC_DIR}/utils/unittest/googletest/include)

add_executable(LLVMFuzzer-Unittest
  FuzzerUnittest.cpp
  $<TARGET_OBJECTS:LLVMFuzzerNoMainObjects>
  )

target_link_libraries(LLVMFuzzer-Unittest
  gtest
  gtest_main
  )

set(TestBinaries ${TestBinaries} LLVMFuzzer-Unittest)

add_subdirectory(dfsan)

foreach(Test ${DFSanTests})
  set(TestBinaries ${TestBinaries} LLVMFuzzer-${Test}-DFSan)
endforeach()

add_subdirectory(uninstrumented)

foreach(Test ${UninstrumentedTests})
  set(TestBinaries ${TestBinaries} LLVMFuzzer-${Test}-Uninstrumented)
endforeach()

add_subdirectory(trace-bb)

foreach(Test ${TraceBBTests})
  set(TestBinaries ${TestBinaries} LLVMFuzzer-${Test}-TraceBB)
endforeach()

set_target_properties(${TestBinaries}
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )

add_lit_testsuite(check-fuzzer "Running Fuzzer tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${TestBinaries} FileCheck not
    )
