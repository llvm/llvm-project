# These tests are not instrumented with coverage and don't
# have coverage rt in the binary.

set(CMAKE_CXX_FLAGS
  "${LIBFUZZER_FLAGS_BASE} -fno-sanitize-coverage=edge,trace-cmp,indirect-calls,8bit-counters,trace-pc-guard -fsanitize-coverage=trace-pc")

set(TracePCTests
  SimpleTest
  )

foreach(Test ${TracePCTests})
  add_libfuzzer_test(${Test}-TracePC SOURCES ../${Test}.cpp)
endforeach()
