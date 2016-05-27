# These tests are not instrumented with coverage.

set(CMAKE_CXX_FLAGS
  "${LIBFUZZER_FLAGS_BASE} -fno-sanitize=all -fno-sanitize-coverage=edge,trace-cmp,indirect-calls,8bit-counters")

set(UninstrumentedTests
  UninstrumentedTest
  )

foreach(Test ${UninstrumentedTests})
  add_libfuzzer_test(${Test}-Uninstrumented SOURCES ../${Test}.cpp)
endforeach()

# Propagate value into parent directory
set(TestBinaries ${TestBinaries} PARENT_SCOPE)
