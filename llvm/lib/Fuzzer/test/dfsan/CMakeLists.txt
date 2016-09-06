# These tests depend on both coverage and dfsan instrumentation.

set(CMAKE_CXX_FLAGS
  "${LIBFUZZER_FLAGS_BASE} -fno-sanitize=all -fsanitize=dataflow")

set(DFSanTests
  MemcmpTest
  SimpleCmpTest
  StrcmpTest
  StrncmpTest
  SwitchTest
  )

foreach(Test ${DFSanTests})
  add_libfuzzer_test(${Test}-DFSan SOURCES ../${Test}.cpp)
endforeach()

# Propagate value into parent directory
set(TestBinaries ${TestBinaries} PARENT_SCOPE)
