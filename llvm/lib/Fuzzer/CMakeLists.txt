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

if (CMAKE_SYSTEM_NAME MATCHES "Darwin|Linux")
  set(LIBFUZZER_ENABLED_CHECK ON)
else()
  set(LIBFUZZER_ENABLED_CHECK OFF)
endif()

# Compile libFuzzer if the compilation is specifically requested, OR
# if the platform is known to be working.
set(LIBFUZZER_ENABLE ${LIBFUZZER_ENABLED_CHECK} CACHE BOOL "Build libFuzzer and its tests")
set(LIBFUZZER_ENABLE_TESTS OFF CACHE BOOL "Build libFuzzer and its tests")

if (LLVM_USE_SANITIZE_COVERAGE)
  set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -fno-sanitize-coverage=trace-pc-guard,edge,trace-cmp,indirect-calls,8bit-counters -Werror")
endif()

if (LIBFUZZER_ENABLE)
  add_library(LLVMFuzzerNoMainObjects OBJECT
      FuzzerClangCounters.cpp
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

if (MSVC)

  # Until bots are reconfigured, check-fuzzer on Windows is a no-OP.
  add_custom_target(check-fuzzer)
  add_custom_command(TARGET check-fuzzer
    COMMAND cmake -E echo "check-fuzzer is disalbed on Windows")
else()
  if (LLVM_INCLUDE_TESTS AND LIBFUZZER_ENABLE_TESTS)
    add_subdirectory(test)
  endif()
endif()
