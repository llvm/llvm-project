include(CheckCXXSourceCompiles)

CHECK_CXX_SOURCE_COMPILES("
    static thread_local int blah;
    int main() {
      return 0;
    }
    " HAS_THREAD_LOCAL)

if( NOT HAS_THREAD_LOCAL )
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Dthread_local=__thread")
endif()

if( LLVM_USE_SANITIZE_COVERAGE )
  set(LIBFUZZER_FLAGS_BASE "${CMAKE_CXX_FLAGS}")

  # Disable the coverage and sanitizer instrumentation for the fuzzer itself.
  set(CMAKE_CXX_FLAGS "${LIBFUZZER_FLAGS_BASE} -fno-sanitize-coverage=trace-pc-guard,edge,trace-cmp,indirect-calls,8bit-counters -Werror")
endif()

add_library(LLVMFuzzerNoMainObjects OBJECT
    FuzzerCrossOver.cpp
    FuzzerDriver.cpp
    FuzzerExtFunctionsDlsym.cpp
    FuzzerExtFunctionsDlsymWin.cpp
    FuzzerExtFunctionsWeak.cpp
    FuzzerExtraCounters.cpp
    FuzzerIO.cpp
    FuzzerIOPosix.cpp
    FuzzerIOWindows.cpp
    FuzzerLoop.cpp
    FuzzerMerge.cpp
    FuzzerMutate.cpp
    FuzzerSHA1.cpp
    FuzzerShmemPosix.cpp
    FuzzerShmemWindows.cpp
    FuzzerTracePC.cpp
    FuzzerTraceState.cpp
    FuzzerUtil.cpp
    FuzzerUtilDarwin.cpp
    FuzzerUtilLinux.cpp
    FuzzerUtilPosix.cpp
    FuzzerUtilWindows.cpp
    )
add_library(LLVMFuzzerNoMain STATIC
    $<TARGET_OBJECTS:LLVMFuzzerNoMainObjects>
    )
target_link_libraries(LLVMFuzzerNoMain ${LLVM_PTHREAD_LIB})
add_library(LLVMFuzzer STATIC
    FuzzerMain.cpp
    $<TARGET_OBJECTS:LLVMFuzzerNoMainObjects>
    )
target_link_libraries(LLVMFuzzer ${LLVM_PTHREAD_LIB})

if( LLVM_USE_SANITIZE_COVERAGE )

  if(NOT "${LLVM_USE_SANITIZER}" STREQUAL "Address")
    message(FATAL_ERROR
      "LibFuzzer tests require LLVM_USE_SANITIZER=Address and "
      "LLVM_USE_SANITIZE_COVERAGE=YES to be set."
      )
  endif()

  if( LLVM_INCLUDE_TESTS )
    add_subdirectory(test)
  endif()
endif()
