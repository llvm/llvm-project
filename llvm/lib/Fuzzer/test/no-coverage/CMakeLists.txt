# These tests are not instrumented with coverage,
# but have coverage rt in the binary.

set(CMAKE_CXX_FLAGS
  "${LIBFUZZER_FLAGS_BASE} -fno-sanitize-coverage=edge,trace-cmp,indirect-calls,8bit-counters")

set(NoCoverageTests
  UninstrumentedTest
  )

foreach(Test ${NoCoverageTests})
  add_libfuzzer_test(${Test}-NoCoverage SOURCES ../${Test}.cpp)
endforeach()

# Propagate value into parent directory
set(TestBinaries ${TestBinaries} PARENT_SCOPE)
