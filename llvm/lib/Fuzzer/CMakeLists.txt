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

if (CMAKE_CXX_FLAGS MATCHES "fsanitize-coverage")
  set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -fsanitize-coverage=0")
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

  if(LLVMFuzzer IN_LIST LLVM_DISTRIBUTION_COMPONENTS OR
      NOT LLVM_DISTRIBUTION_COMPONENTS)
    set(export_to_llvmexports EXPORT LLVMExports)
    set_property(GLOBAL PROPERTY LLVM_HAS_EXPORTS True)
  endif()

  install(TARGETS LLVMFuzzer
          ${export_to_llvmexports}
          ARCHIVE DESTINATION lib${LLVM_LIBDIR_SUFFIX}
          COMPONENT LLVMFuzzer)

  if (NOT CMAKE_CONFIGURATION_TYPES)
    add_custom_target(install-LLVMFuzzer
                      DEPENDS LLVMFuzzer
                      COMMAND "${CMAKE_COMMAND}"
                              -DCMAKE_INSTALL_COMPONENT=LLVMFuzzer
                              -P "${CMAKE_BINARY_DIR}/cmake_install.cmake")
  endif()

  set_property(GLOBAL APPEND PROPERTY LLVM_EXPORTS LLVMFuzzer)
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
