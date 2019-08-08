# Check if the host compiler is new enough.
# These versions are updated based on the following policy:
#   llvm.org/docs/DeveloperPolicy.html#toolchain

include(CheckCXXSourceCompiles)

set(GCC_MIN 4.8)
set(GCC_SOFT_ERROR 5.1)
set(CLANG_MIN 3.1)
set(CLANG_SOFT_ERROR 3.5)
set(APPLECLANG_MIN 3.1)
set(APPLECLANG_SOFT_ERROR 6.0)

# https://en.wikipedia.org/wiki/Microsoft_Visual_C#Internal_version_numbering
# _MSC_VER == 1910 MSVC++ 14.1 (Visual Studio 2017 version 15.0)
set(MSVC_MIN 19.1)
set(MSVC_SOFT_ERROR 19.1)

# Map the above GCC versions to dates: https://gcc.gnu.org/develop.html#timeline
set(GCC_MIN_DATE 20130322)
set(GCC_SOFT_ERROR_DATE 20150422)


if(DEFINED LLVM_COMPILER_CHECKED)
  return()
endif()
set(LLVM_COMPILER_CHECKED ON)

if(LLVM_FORCE_USE_OLD_TOOLCHAIN)
  return()
endif()

function(check_compiler_version NAME NICE_NAME MINIMUM_VERSION SOFT_ERROR_VERSION)
  if(NOT CMAKE_CXX_COMPILER_ID STREQUAL NAME)
    return()
  endif()
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS MINIMUM_VERSION)
    message(FATAL_ERROR "Host ${NICE_NAME} version must be at least ${MINIMUM_VERSION}, your version is ${CMAKE_CXX_COMPILER_VERSION}.")
  elseif(CMAKE_CXX_COMPILER_VERSION VERSION_LESS SOFT_ERROR_VERSION)
    if(LLVM_TEMPORARILY_ALLOW_OLD_TOOLCHAIN)
      message(WARNING "Host ${NICE_NAME} version should be at least ${SOFT_ERROR_VERSION} because LLVM will soon use new C++ features which your toolchain version doesn't support. Your version is ${CMAKE_CXX_COMPILER_VERSION}. Ignoring because you've set LLVM_TEMPORARILY_ALLOW_OLD_TOOLCHAIN, but very soon your toolchain won't be supported.")
    else()
      message(FATAL_ERROR "Host ${NICE_NAME} version should be at least ${SOFT_ERROR_VERSION} because LLVM will soon use new C++ features which your toolchain version doesn't support. Your version is ${CMAKE_CXX_COMPILER_VERSION}. You can temporarily opt out using LLVM_TEMPORARILY_ALLOW_OLD_TOOLCHAIN, but very soon your toolchain won't be supported.")
    endif()
  endif()
endfunction(check_compiler_version)

check_compiler_version("GNU" "GCC" ${GCC_MIN} ${GCC_SOFT_ERROR})
check_compiler_version("Clang" "Clang" ${CLANG_MIN} ${CLANG_SOFT_ERROR})
check_compiler_version("AppleClang" "Apple Clang" ${APPLECLANG_MIN} ${APPLECLANG_SOFT_ERROR})
check_compiler_version("MSVC" "Visual Studio" ${MSVC_MIN} ${MSVC_SOFT_ERROR})

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  if (CMAKE_CXX_SIMULATE_ID MATCHES "MSVC")
    if (CMAKE_CXX_SIMULATE_VERSION VERSION_LESS MSVC_MIN)
      message(FATAL_ERROR "Host Clang must have at least -fms-compatibility-version=${MSVC_MIN}, your version is ${CMAKE_CXX_SIMULATE_VERSION}.")
    endif()
    set(CLANG_CL 1)
  elseif(NOT LLVM_ENABLE_LIBCXX)
    # Test that we aren't using too old of a version of libstdc++.
    set(OLD_CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS})
    set(OLD_CMAKE_REQUIRED_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES})
    set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} -std=c++0x")
    # Test for libstdc++ version of at least 4.8 by checking for _ZNKSt17bad_function_call4whatEv.
    # Note: We should check _GLIBCXX_RELEASE when possible (i.e., for GCC 7.1 and up).
    check_cxx_source_compiles("
#include <iosfwd>
#if defined(__GLIBCXX__)
#if __GLIBCXX__ < ${GCC_MIN_DATE}
#error Unsupported libstdc++ version
#endif
#endif
#if defined(__GLIBCXX__)
extern const char _ZNKSt17bad_function_call4whatEv[];
const char *chk = _ZNKSt17bad_function_call4whatEv;
#else
const char *chk = \"\";
#endif
int main() { ++chk; return 0; }
"
      LLVM_LIBSTDCXX_MIN)
    if(NOT LLVM_LIBSTDCXX_MIN)
      message(FATAL_ERROR "libstdc++ version must be at least ${GCC_MIN}.")
    endif()
    # Test for libstdc++ version of at least 5.1 by checking for std::iostream_category().
    # Note: We should check _GLIBCXX_RELEASE when possible (i.e., for GCC 7.1 and up).
    check_cxx_source_compiles("
#include <iosfwd>
#if defined(__GLIBCXX__)
#if __GLIBCXX__ < ${GCC_SOFT_ERROR_DATE}
#error Unsupported libstdc++ version
#endif
#endif
#if defined(__GLIBCXX__)
#include <ios>
void foo(void) { (void) std::iostream_category(); }
#endif
int main() { return 0; }
"
      LLVM_LIBSTDCXX_SOFT_ERROR)
    if(NOT LLVM_LIBSTDCXX_SOFT_ERROR)
      if(LLVM_TEMPORARILY_ALLOW_OLD_TOOLCHAIN)
        message(WARNING "libstdc++ version should be at least ${GCC_SOFT_ERROR} because LLVM will soon use new C++ features which your toolchain version doesn't support. Ignoring because you've set LLVM_TEMPORARILY_ALLOW_OLD_TOOLCHAIN, but very soon your toolchain won't be supported.")
      else()
        message(FATAL_ERROR "libstdc++ version should be at least ${GCC_SOFT_ERROR} because LLVM will soon use new C++ features which your toolchain version doesn't support. You can temporarily opt out using LLVM_TEMPORARILY_ALLOW_OLD_TOOLCHAIN, but very soon your toolchain won't be supported.")
      endif()
    endif()
    set(CMAKE_REQUIRED_FLAGS ${OLD_CMAKE_REQUIRED_FLAGS})
    set(CMAKE_REQUIRED_LIBRARIES ${OLD_CMAKE_REQUIRED_LIBRARIES})
  endif()
endif()
