include(CheckCXXCompilerFlag)
include(CheckCXXSourceCompiles)

# Check for oneAPI compiler (some older CMake versions detect as Clang)
if (CMAKE_C_COMPILER_ID STREQUAL "Clang")
  check_cxx_source_compiles("#if (defined(__INTEL_CLANG_COMPILER) || defined(__INTEL_LLVM_COMPILER))
                             int main() { return 0; }
                             #else
                             not oneAPI
                             #endif" OPENMP_HAVE_ONEAPI_COMPILER)
  if (OPENMP_HAVE_ONEAPI_COMPILER)
    # According to CMake documentation, the compiler id should
    # be IntelLLVM when detected oneAPI
    set(CMAKE_C_COMPILER_ID "IntelLLVM")
    set(CMAKE_CXX_COMPILER_ID "IntelLLVM")
  endif()
endif()

check_cxx_compiler_flag(-Wall OPENMP_HAVE_WALL_FLAG)
check_cxx_compiler_flag(-Werror OPENMP_HAVE_WERROR_FLAG)

# Additional warnings that are not enabled by -Wall.
check_cxx_compiler_flag(-Wcast-qual OPENMP_HAVE_WCAST_QUAL_FLAG)
check_cxx_compiler_flag(-Wformat-pedantic OPENMP_HAVE_WFORMAT_PEDANTIC_FLAG)
check_cxx_compiler_flag(-Wimplicit-fallthrough OPENMP_HAVE_WIMPLICIT_FALLTHROUGH_FLAG)
check_cxx_compiler_flag(-Wsign-compare OPENMP_HAVE_WSIGN_COMPARE_FLAG)

# Warnings that we want to disable because they are too verbose or fragile.
check_cxx_compiler_flag(-Wno-extra OPENMP_HAVE_WNO_EXTRA_FLAG)
check_cxx_compiler_flag(-Wno-pedantic OPENMP_HAVE_WNO_PEDANTIC_FLAG)
check_cxx_compiler_flag(-Wno-maybe-uninitialized OPENMP_HAVE_WNO_MAYBE_UNINITIALIZED_FLAG)

check_cxx_compiler_flag(-std=c++14 OPENMP_HAVE_STD_CPP14_FLAG)
