# These tests are not instrumented with coverage.

set(CMAKE_CXX_FLAGS_RELEASE
  "${LIBFUZZER_FLAGS_BASE} -fsanitize-coverage=edge,trace-bb")

foreach(Test ${TraceBBTests})
  add_executable(LLVMFuzzer-${Test}-TraceBB
    ../${Test}.cpp
    )
  target_link_libraries(LLVMFuzzer-${Test}-TraceBB
    LLVMFuzzer
    )
endforeach()

