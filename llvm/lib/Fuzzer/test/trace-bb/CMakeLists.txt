# These tests are not instrumented with coverage.

set(CMAKE_CXX_FLAGS
  "${LIBFUZZER_FLAGS_BASE} -fsanitize-coverage=edge,trace-bb")

set(TraceBBTests
  SimpleTest
  )

foreach(Test ${TraceBBTests})
  add_libfuzzer_test(${Test}-TraceBB SOURCES ../${Test}.cpp)
endforeach()

# Propagate value into parent directory
set(TestBinaries ${TestBinaries} PARENT_SCOPE)
