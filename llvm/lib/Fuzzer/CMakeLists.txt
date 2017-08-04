include(CheckCXXSourceCompiles)

if( APPLE )
  CHECK_CXX_SOURCE_COMPILES("
      static thread_local int blah;
      int main() {
        return 0;
      }
      " HAS_THREAD_LOCAL)

  if( NOT HAS_THREAD_LOCAL )
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Dthread_local=__thread")
  endif()
endif()

set(LIBFUZZER_FLAGS_BASE "${CMAKE_CXX_FLAGS}")
if( LLVM_USE_SANITIZE_COVERAGE )
  if(NOT "${LLVM_USE_SANITIZER}" STREQUAL "Address")
    message(FATAL_ERROR
      "LibFuzzer and its tests require LLVM_USE_SANITIZER=Address and "
      "LLVM_USE_SANITIZE_COVERAGE=YES to be set."
      )
  endif()

  # Disable the coverage and sanitizer instrumentation for the fuzzer itself.
  set(CMAKE_CXX_FLAGS "${LIBFUZZER_FLAGS_BASE} -fno-sanitize-coverage=trace-pc-guard,edge,trace-cmp,indirect-calls,8bit-counters -Werror")
endif()

# Compile libFuzzer if the compilation is specifically requested, OR
# if the platform is known to be working.
if ( LLVM_USE_SANITIZE_COVERAGE OR CMAKE_SYSTEM_NAME MATCHES "Darwin|Linux" )
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
endif()

if( LLVM_USE_SANITIZE_COVERAGE AND LLVM_INCLUDE_TESTS )
  add_subdirectory(test)
endif()
