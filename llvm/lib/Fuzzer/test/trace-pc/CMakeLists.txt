# These tests are instrumented with -fsanitize-coverage=trace-pc

set(CMAKE_CXX_FLAGS
  "${LIBFUZZER_FLAGS_BASE} -fno-sanitize-coverage=edge,trace-cmp,indirect-calls,8bit-counters,trace-pc-guard -fsanitize-coverage=trace-pc")

set(TracePCTests
  SimpleTest
  )

foreach(Test ${TracePCTests})
  add_libfuzzer_test(${Test}-TracePC SOURCES ../${Test}.cpp)
endforeach()
