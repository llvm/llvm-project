# Check if the host compiler is new enough.
# These versions are updated based on the following policy:
#   llvm.org/docs/DeveloperPolicy.html#toolchain

include(CheckCXXSourceCompiles)

set(GCC_MIN 4.8)
set(GCC_WARN 4.8)
set(CLANG_MIN 3.1)
set(CLANG_WARN 3.1)
set(APPLECLANG_MIN 3.1)
set(APPLECLANG_WARN 3.1)
set(MSVC_MIN 19.0)
set(MSVC_WARN 19.00.24213.1)

if(DEFINED LLVM_COMPILER_CHECKED)
  return()
endif()
set(LLVM_COMPILER_CHECKED ON)

if(LLVM_FORCE_USE_OLD_TOOLCHAIN)
  return()
endif()

function(check_compiler_version NAME NICE_NAME MINIMUM_VERSION WARN_VERSION)
  if(NOT CMAKE_CXX_COMPILER_ID STREQUAL NAME)
    return()
  endif()
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS MINIMUM_VERSION)
    message(FATAL_ERROR "Host ${NICE_NAME} version must be at least ${MINIMUM_VERSION}, your version is ${CMAKE_CXX_COMPILER_VERSION}.")
  elseif(CMAKE_CXX_COMPILER_VERSION VERSION_LESS WARN_VERSION)
    message(WARNING "Host ${NICE_NAME} version must be at least ${WARN_VERSION} due to miscompiles from earlier versions, your version is ${CMAKE_CXX_COMPILER_VERSION}.")
  endif()
endfunction(check_compiler_version)

check_compiler_version("GNU" "GCC" ${GCC_MIN} ${GCC_WARN})
check_compiler_version("Clang" "Clang" ${CLANG_MIN} ${CLANG_WARN})
check_compiler_version("AppleClang" "Apple Clang" ${APPLECLANG_MIN} ${APPLECLANG_WARN})
check_compiler_version("MSVC" "Visual Studio" ${MSVC_MIN} ${MSVC_WARN})

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  if (CMAKE_CXX_SIMULATE_ID MATCHES "MSVC")
    if (CMAKE_CXX_SIMULATE_VERSION VERSION_LESS MSVC_MIN)
      message(FATAL_ERROR "Host Clang must have at least -fms-compatibility-version=${MSVC_MIN}, your version is ${CMAKE_CXX_COMPILER_VERSION}.")
    endif()
    set(CLANG_CL 1)
  elseif(NOT LLVM_ENABLE_LIBCXX)
    # Test that we aren't using too old of a version of libstdc++
    # with the Clang compiler. This is tricky as there is no real way to
    # check the version of libstdc++ directly. Instead we test for a known
    # bug in libstdc++4.6 that is fixed in libstdc++4.7.
    set(OLD_CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS})
    set(OLD_CMAKE_REQUIRED_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES})
    set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} -std=c++0x")
    check_cxx_source_compiles("
#include <atomic>
std::atomic<float> x(0.0f);
int main() { return (float)x; }"
      LLVM_NO_OLD_LIBSTDCXX)
    if(NOT LLVM_NO_OLD_LIBSTDCXX)
      message(FATAL_ERROR "Host Clang must be able to find libstdc++4.8 or newer!")
    endif()
    set(CMAKE_REQUIRED_FLAGS ${OLD_CMAKE_REQUIRED_FLAGS})
    set(CMAKE_REQUIRED_LIBRARIES ${OLD_CMAKE_REQUIRED_LIBRARIES})
  endif()
endif()
