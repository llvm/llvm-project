# These tests are not instrumented with coverage.

set(CMAKE_CXX_FLAGS
  "${LIBFUZZER_FLAGS_BASE} -fno-sanitize-coverage=8bit-counters -fsanitize-coverage=trace-pc-guard")

set(TracePCTests
  SimpleTest
  CounterTest
  CallerCalleeTest
  )

foreach(Test ${TracePCTests})
  add_libfuzzer_test(${Test}-TracePC SOURCES ../${Test}.cpp)
endforeach()

# Propagate value into parent directory
set(TestBinaries ${TestBinaries} PARENT_SCOPE)
