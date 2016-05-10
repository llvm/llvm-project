# These tests are instrumented with ubsan in non-recovery mode.

set(CMAKE_CXX_FLAGS_RELEASE
  "${LIBFUZZER_FLAGS_BASE} -O0 -fsanitize=undefined -fno-sanitize-recover=all")

foreach(Test ${UbsanTests})
  add_executable(LLVMFuzzer-${Test}-Ubsan
    ../${Test}.cpp
    )
  target_link_libraries(LLVMFuzzer-${Test}-Ubsan
    LLVMFuzzer
    )
endforeach()

