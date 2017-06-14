# These tests are instrumented with -fsanitize-coverage=inline-8bit-counters

set(CMAKE_CXX_FLAGS
  "${LIBFUZZER_FLAGS_BASE} -fno-sanitize-coverage=trace-pc-guard -fsanitize-coverage=inline-8bit-counters")

set(Inline8bitCounterTests
  SimpleTest
  )

foreach(Test ${Inline8bitCounterTests})
  add_libfuzzer_test(${Test}-Inline8bitCounters SOURCES ../${Test}.cpp)
endforeach()
