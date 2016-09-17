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

add_library(LLVMFuzzer-DSO1 SHARED ../DSO1.cpp)
add_library(LLVMFuzzer-DSO2 SHARED ../DSO2.cpp)

add_executable(LLVMFuzzer-DSOTest
  ../DSOTestMain.cpp
  ../DSOTestExtra.cpp)

target_link_libraries(LLVMFuzzer-DSOTest
  LLVMFuzzer-DSO1
  LLVMFuzzer-DSO2
  LLVMFuzzer
  )

set_target_properties(LLVMFuzzer-DSOTest PROPERTIES RUNTIME_OUTPUT_DIRECTORY
  "${CMAKE_BINARY_DIR}/lib/Fuzzer/test")
set_target_properties(LLVMFuzzer-DSO1 PROPERTIES LIBRARY_OUTPUT_DIRECTORY
  "${CMAKE_BINARY_DIR}/lib/Fuzzer/lib")
set_target_properties(LLVMFuzzer-DSO2 PROPERTIES LIBRARY_OUTPUT_DIRECTORY
  "${CMAKE_BINARY_DIR}/lib/Fuzzer/lib")

set(TestBinaries ${TestBinaries} LLVMFuzzer-DSOTest PARENT_SCOPE)
