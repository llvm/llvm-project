set(LIBFUZZER_FLAGS_BASE "${CMAKE_CXX_FLAGS_RELEASE}")
# Disable the coverage and sanitizer instrumentation for the fuzzer itself.
set(CMAKE_CXX_FLAGS_RELEASE "${LIBFUZZER_FLAGS_BASE} -O2 -fno-sanitize=all -fno-sanitize-coverage=edge,trace-cmp,indirect-calls,8bit-counters -Werror")
if( LLVM_USE_SANITIZE_COVERAGE )
  add_library(LLVMFuzzerNoMainObjects OBJECT
    FuzzerCrossOver.cpp
    FuzzerInterface.cpp
    FuzzerTraceState.cpp
    FuzzerDriver.cpp
    FuzzerIO.cpp
    FuzzerLoop.cpp
    FuzzerMutate.cpp
    FuzzerSHA1.cpp
    FuzzerTracePC.cpp
    FuzzerUtil.cpp
    )
  add_library(LLVMFuzzerNoMain STATIC
    $<TARGET_OBJECTS:LLVMFuzzerNoMainObjects>
    )
  target_link_libraries(LLVMFuzzerNoMain ${PTHREAD_LIB})
  add_library(LLVMFuzzer STATIC
    FuzzerMain.cpp
    $<TARGET_OBJECTS:LLVMFuzzerNoMainObjects>
    )
  target_link_libraries(LLVMFuzzer ${PTHREAD_LIB})

  if( LLVM_INCLUDE_TESTS )
    add_subdirectory(test)
  endif()
endif()
