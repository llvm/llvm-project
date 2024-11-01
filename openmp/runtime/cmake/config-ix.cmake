#
#//===----------------------------------------------------------------------===//
#//
#// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#// See https://llvm.org/LICENSE.txt for license information.
#// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#//
#//===----------------------------------------------------------------------===//
#

include(CheckCCompilerFlag)
include(CheckCSourceCompiles)
include(CheckCXXSourceCompiles)
include(CheckCXXCompilerFlag)
include(CheckIncludeFile)
include(CheckLibraryExists)
include(CheckIncludeFiles)
include(CheckSymbolExists)
include(LibompCheckFortranFlag)
include(LLVMCheckCompilerLinkerFlag)

# Check for versioned symbols
function(libomp_check_version_symbols retval)
  set(source_code
    "#include <stdio.h>
    void func1() { printf(\"Hello\"); }
    void func2() { printf(\"World\"); }
    __asm__(\".symver func1, func@VER1\");
    __asm__(\".symver func2, func@VER2\");
    int main(void) {
      func1();
      func2();
      return 0;
    }")
  set(version_script_source "VER1 { }; VER2 { } VER1;")
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/__version_script.txt "${version_script_source}")
  set(CMAKE_REQUIRED_FLAGS -Wl,--version-script=${CMAKE_CURRENT_BINARY_DIR}/__version_script.txt)
  check_c_source_compiles("${source_code}" ${retval})
  set(${retval} ${${retval}} PARENT_SCOPE)
  file(REMOVE ${CMAKE_CURRENT_BINARY_DIR}/__version_script.txt)
endfunction()

# Includes the architecture flag in both compile and link phase
function(libomp_check_architecture_flag flag retval)
  set(CMAKE_REQUIRED_FLAGS "${flag}")
  check_c_compiler_flag("${flag}" ${retval})
  set(${retval} ${${retval}} PARENT_SCOPE)
endfunction()

# Checking CXX, Linker Flags

# GCC silently accepts any -Wno-<foo> option, but warns about those options
# being unrecognized only if the compilation triggers other warnings to be
# printed. Therefore, check for whether the compiler supports options in the
# form -W<foo>, and if supported, add the corresponding -Wno-<foo> option.

check_cxx_compiler_flag(-fno-exceptions LIBOMP_HAVE_FNO_EXCEPTIONS_FLAG)
check_cxx_compiler_flag(-fno-rtti LIBOMP_HAVE_FNO_RTTI_FLAG)
check_cxx_compiler_flag(-Wclass-memaccess LIBOMP_HAVE_WCLASS_MEMACCESS_FLAG)
check_cxx_compiler_flag(-Wcovered-switch-default LIBOMP_HAVE_WCOVERED_SWITCH_DEFAULT_FLAG)
check_cxx_compiler_flag(-Wframe-address LIBOMP_HAVE_WFRAME_ADDRESS_FLAG)
check_cxx_compiler_flag(-Wstrict-aliasing LIBOMP_HAVE_WSTRICT_ALIASING_FLAG)
check_cxx_compiler_flag(-Wstringop-overflow=0 LIBOMP_HAVE_WSTRINGOP_OVERFLOW_FLAG)
check_cxx_compiler_flag(-Wstringop-truncation LIBOMP_HAVE_WSTRINGOP_TRUNCATION_FLAG)
check_cxx_compiler_flag(-Wswitch LIBOMP_HAVE_WSWITCH_FLAG)
check_cxx_compiler_flag(-Wuninitialized LIBOMP_HAVE_WUNINITIALIZED_FLAG)
check_cxx_compiler_flag(-Wreturn-type-c-linkage LIBOMP_HAVE_WRETURN_TYPE_C_LINKAGE_FLAG)
check_cxx_compiler_flag(-Wcast-qual LIBOMP_HAVE_WCAST_QUAL_FLAG)
check_cxx_compiler_flag(-Wint-to-void-pointer-cast LIBOMP_HAVE_WINT_TO_VOID_POINTER_CAST_FLAG)
# check_cxx_compiler_flag(-Wconversion LIBOMP_HAVE_WCONVERSION_FLAG)
check_cxx_compiler_flag(-msse2 LIBOMP_HAVE_MSSE2_FLAG)
check_cxx_compiler_flag(-ftls-model=initial-exec LIBOMP_HAVE_FTLS_MODEL_FLAG)
libomp_check_architecture_flag(-mmic LIBOMP_HAVE_MMIC_FLAG)
libomp_check_architecture_flag(-m32 LIBOMP_HAVE_M32_FLAG)
if(WIN32)
  if(MSVC)
    # Check Windows MSVC style flags.
    check_cxx_compiler_flag(/EHsc LIBOMP_HAVE_EHSC_FLAG)
    check_cxx_compiler_flag(/GS LIBOMP_HAVE_GS_FLAG)
    check_cxx_compiler_flag(/Oy- LIBOMP_HAVE_Oy__FLAG)
    check_cxx_compiler_flag(/arch:SSE2 LIBOMP_HAVE_ARCH_SSE2_FLAG)
    check_cxx_compiler_flag(/Qsafeseh LIBOMP_HAVE_QSAFESEH_FLAG)
  endif()
  check_cxx_compiler_flag(-mrtm LIBOMP_HAVE_MRTM_FLAG)
  # It is difficult to create a dummy masm assembly file
  # and then check the MASM assembler to see if these flags exist and work,
  # so we assume they do for Windows.
  set(LIBOMP_HAVE_SAFESEH_MASM_FLAG TRUE)
  set(LIBOMP_HAVE_COFF_MASM_FLAG TRUE)
  # Change Windows flags /MDx to /MTx
  foreach(libomp_lang IN ITEMS C CXX)
    foreach(libomp_btype IN ITEMS DEBUG RELWITHDEBINFO RELEASE MINSIZEREL)
      string(REPLACE "/MD" "/MT"
        CMAKE_${libomp_lang}_FLAGS_${libomp_btype}
        "${CMAKE_${libomp_lang}_FLAGS_${libomp_btype}}"
      )
    endforeach()
  endforeach()
endif()
if(${LIBOMP_FORTRAN_MODULES})
  libomp_check_fortran_flag(-m32 LIBOMP_HAVE_M32_FORTRAN_FLAG)
endif()

# Check for Unix shared memory
check_symbol_exists(shm_open "sys/mman.h" LIBOMP_HAVE_SHM_OPEN_NO_LRT)
if (NOT LIBOMP_HAVE_SHM_OPEN_NO_LRT)
  set(CMAKE_REQUIRED_LIBRARIES -lrt)
  check_symbol_exists(shm_open "sys/mman.h" LIBOMP_HAVE_SHM_OPEN_WITH_LRT)
  set(CMAKE_REQUIRED_LIBRARIES)
endif()

# Check for aligned memory allocator function
check_include_file(xmmintrin.h LIBOMP_HAVE_XMMINTRIN_H)
set(OLD_CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS})
if (LIBOMP_HAVE_XMMINTRIN_H)
  set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} -DLIBOMP_HAVE_XMMINTRIN_H")
endif()
set(source_code "// check for _mm_malloc
    #ifdef LIBOMP_HAVE_XMMINTRIN_H
    #include <xmmintrin.h>
    #endif
    int main() { void *ptr = _mm_malloc(sizeof(int) * 1000, 64); _mm_free(ptr); return 0; }")
check_cxx_source_compiles("${source_code}" LIBOMP_HAVE__MM_MALLOC)
set(CMAKE_REQUIRED_FLAGS ${OLD_CMAKE_REQUIRED_FLAGS})
check_symbol_exists(aligned_alloc "stdlib.h" LIBOMP_HAVE_ALIGNED_ALLOC)
check_symbol_exists(posix_memalign "stdlib.h" LIBOMP_HAVE_POSIX_MEMALIGN)
check_symbol_exists(_aligned_malloc "malloc.h" LIBOMP_HAVE__ALIGNED_MALLOC)

# Check linker flags
if(WIN32)
  llvm_check_compiler_linker_flag(C /SAFESEH LIBOMP_HAVE_SAFESEH_FLAG)
elseif(NOT APPLE)
  llvm_check_compiler_linker_flag(C -Wl,-x LIBOMP_HAVE_X_FLAG)
  llvm_check_compiler_linker_flag(C -Wl,--as-needed LIBOMP_HAVE_AS_NEEDED_FLAG)
  llvm_check_compiler_linker_flag(C "-Wl,--version-script=${LIBOMP_SRC_DIR}/exports_test_so.txt" LIBOMP_HAVE_VERSION_SCRIPT_FLAG)
  llvm_check_compiler_linker_flag(C -static-libgcc LIBOMP_HAVE_STATIC_LIBGCC_FLAG)
  llvm_check_compiler_linker_flag(C -Wl,-z,noexecstack LIBOMP_HAVE_Z_NOEXECSTACK_FLAG)
endif()

# Check Intel(R) C Compiler specific flags
if(CMAKE_C_COMPILER_ID STREQUAL "Intel" OR CMAKE_C_COMPILER_ID STREQUAL "IntelLLVM")
  check_cxx_compiler_flag(/Qlong_double LIBOMP_HAVE_LONG_DOUBLE_FLAG)
  check_cxx_compiler_flag(/Qdiag-disable:177 LIBOMP_HAVE_DIAG_DISABLE_177_FLAG)
  check_cxx_compiler_flag(/Qinline-min-size=1 LIBOMP_HAVE_INLINE_MIN_SIZE_FLAG)
  check_cxx_compiler_flag(-Qoption,cpp,--extended_float_types LIBOMP_HAVE_EXTENDED_FLOAT_TYPES_FLAG)
  check_cxx_compiler_flag(-falign-stack=maintain-16-byte LIBOMP_HAVE_FALIGN_STACK_FLAG)
  check_cxx_compiler_flag("-opt-streaming-stores never" LIBOMP_HAVE_OPT_STREAMING_STORES_FLAG)
  llvm_check_compiler_linker_flag(C -static-intel LIBOMP_HAVE_STATIC_INTEL_FLAG)
  llvm_check_compiler_linker_flag(C -no-intel-extensions LIBOMP_HAVE_NO_INTEL_EXTENSIONS_FLAG)
  check_library_exists(irc_pic _intel_fast_memcpy "" LIBOMP_HAVE_IRC_PIC_LIBRARY)
endif()

# Checking Threading requirements
find_package(Threads REQUIRED)
if(WIN32)
  if(NOT CMAKE_USE_WIN32_THREADS_INIT)
    libomp_error_say("Need Win32 thread interface on Windows.")
  endif()
else()
  if(NOT CMAKE_USE_PTHREADS_INIT)
    libomp_error_say("Need pthread interface on Unix-like systems.")
  endif()
endif()

# Checking for x86-specific waitpkg and rtm attribute and intrinsics
if (IA32 OR INTEL64)
  check_include_file(immintrin.h LIBOMP_HAVE_IMMINTRIN_H)
  if (NOT LIBOMP_HAVE_IMMINTRIN_H)
    check_include_file(intrin.h LIBOMP_HAVE_INTRIN_H)
  endif()
  check_cxx_source_compiles("__attribute__((target(\"rtm\")))
                             int main() {return 0;}" LIBOMP_HAVE_ATTRIBUTE_RTM)
  check_cxx_source_compiles("__attribute__((target(\"waitpkg\")))
                            int main() {return 0;}" LIBOMP_HAVE_ATTRIBUTE_WAITPKG)
  libomp_append(CMAKE_REQUIRED_DEFINITIONS -DIMMINTRIN_H LIBOMP_HAVE_IMMINTRIN_H)
  libomp_append(CMAKE_REQUIRED_DEFINITIONS -DINTRIN_H LIBOMP_HAVE_INTRIN_H)
  libomp_append(CMAKE_REQUIRED_DEFINITIONS -DATTRIBUTE_WAITPKG LIBOMP_HAVE_ATTRIBUTE_WAITPKG)
  libomp_append(CMAKE_REQUIRED_DEFINITIONS -DATTRIBUTE_RTM LIBOMP_HAVE_ATTRIBUTE_RTM)
  set(source_code "// check for attribute and wait pkg intrinsics
      #ifdef IMMINTRIN_H
      #include <immintrin.h>
      #endif
      #ifdef INTRIN_H
      #include <intrin.h>
      #endif
      #ifdef ATTRIBUTE_WAITPKG
      __attribute__((target(\"waitpkg\")))
      #endif
      static inline int __kmp_umwait(unsigned hint, unsigned long long counter) {
        return _umwait(hint, counter);
      }
      int main() { int a = __kmp_umwait(0, 1000); return a; }")
  check_cxx_source_compiles("${source_code}" LIBOMP_HAVE_WAITPKG_INTRINSICS)
  set(OLD_CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS})
  if (LIBOMP_HAVE_MRTM_FLAG)
    set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} -mrtm")
  endif()
  set(source_code "// check for attribute rtm and rtm intrinsics
      #ifdef IMMINTRIN_H
      #include <immintrin.h>
      #endif
      #ifdef INTRIN_H
      #include <intrin.h>
      #endif
      #ifdef ATTRIBUTE_RTM
      __attribute__((target(\"rtm\")))
      #endif
      static inline int __kmp_xbegin() {
        return _xbegin();
      }
      int main() { int a = __kmp_xbegin(); return a; }")
  check_cxx_source_compiles("${source_code}" LIBOMP_HAVE_RTM_INTRINSICS)
  set(CMAKE_REQUIRED_DEFINITIONS)
  set(CMAKE_REQUIRED_FLAGS ${OLD_CMAKE_REQUIRED_FLAGS})
endif()

# Find perl executable
# Perl is used to create omp.h (and other headers) along with kmp_i18n_id.inc and kmp_i18n_default.inc
find_package(Perl REQUIRED)
# The perl scripts take the --os=/--arch= flags which expect a certain format for operating systems and arch's.
# Until the perl scripts are removed, the most portable way to handle this is to have all operating systems that
# are neither Windows nor Mac (Most Unix flavors) be considered lin to the perl scripts.  This is rooted
# in that all the Perl scripts check the operating system and will fail if it isn't "valid".  This
# temporary solution lets us avoid trying to enumerate all the possible OS values inside the Perl modules.
if(WIN32)
  set(LIBOMP_PERL_SCRIPT_OS win)
elseif(APPLE)
  set(LIBOMP_PERL_SCRIPT_OS mac)
else()
  set(LIBOMP_PERL_SCRIPT_OS lin)
endif()
if(IA32)
  set(LIBOMP_PERL_SCRIPT_ARCH 32)
elseif(MIC)
  set(LIBOMP_PERL_SCRIPT_ARCH mic)
elseif(INTEL64)
  set(LIBOMP_PERL_SCRIPT_ARCH 32e)
else()
  set(LIBOMP_PERL_SCRIPT_ARCH ${LIBOMP_ARCH})
endif()

# Checking features
# Check if version symbol assembler directives are supported
if (LIBOMP_HAVE_VERSION_SCRIPT_FLAG)
  libomp_check_version_symbols(LIBOMP_HAVE_VERSION_SYMBOLS)
else()
  set(LIBOMP_HAVE_VERSION_SYMBOLS FALSE)
endif()

# Check if quad precision types are available
if(CMAKE_C_COMPILER_ID STREQUAL "GNU")
  set(LIBOMP_HAVE_QUAD_PRECISION TRUE)
elseif(CMAKE_C_COMPILER_ID STREQUAL "Intel" OR CMAKE_C_COMPILER_ID STREQUAL "IntelLLVM")
  if(LIBOMP_HAVE_EXTENDED_FLOAT_TYPES_FLAG)
    set(LIBOMP_HAVE_QUAD_PRECISION TRUE)
  else()
    set(LIBOMP_HAVE_QUAD_PRECISION TRUE)
  endif()
else()
  set(LIBOMP_HAVE_QUAD_PRECISION FALSE)
endif()

# Check if adaptive locks are available
if((${IA32} OR ${INTEL64}) AND NOT MSVC)
  set(LIBOMP_HAVE_ADAPTIVE_LOCKS TRUE)
else()
  set(LIBOMP_HAVE_ADAPTIVE_LOCKS FALSE)
endif()

# Check if stats-gathering is available
if(${LIBOMP_STATS})
  check_c_source_compiles(
     "__thread int x;
     int main(int argc, char** argv)
     { x = argc; return x; }"
     LIBOMP_HAVE___THREAD)
  check_c_source_compiles(
     "int main(int argc, char** argv)
     { unsigned long long t = __builtin_readcyclecounter(); return 0; }"
     LIBOMP_HAVE___BUILTIN_READCYCLECOUNTER)
  if(NOT LIBOMP_HAVE___BUILTIN_READCYCLECOUNTER)
    if(${IA32} OR ${INTEL64} OR ${MIC})
      check_include_file(x86intrin.h LIBOMP_HAVE_X86INTRIN_H)
      libomp_append(CMAKE_REQUIRED_DEFINITIONS -DLIBOMP_HAVE_X86INTRIN_H LIBOMP_HAVE_X86INTRIN_H)
      check_c_source_compiles(
        "#ifdef LIBOMP_HAVE_X86INTRIN_H
         # include <x86intrin.h>
         #endif
         int main(int argc, char** argv) { unsigned long long t = __rdtsc(); return 0; }" LIBOMP_HAVE___RDTSC)
      set(CMAKE_REQUIRED_DEFINITIONS)
    endif()
  endif()
  if(LIBOMP_HAVE___THREAD AND (LIBOMP_HAVE___RDTSC OR LIBOMP_HAVE___BUILTIN_READCYCLECOUNTER))
    set(LIBOMP_HAVE_STATS TRUE)
  else()
    set(LIBOMP_HAVE_STATS FALSE)
  endif()
endif()

# Check if OMPT support is available
# Currently, __builtin_frame_address() is required for OMPT
# Weak attribute is required for Unices (except Darwin), LIBPSAPI is used for Windows
check_c_source_compiles("int main(int argc, char** argv) {
  void* p = __builtin_frame_address(0);
  return 0;}" LIBOMP_HAVE___BUILTIN_FRAME_ADDRESS)
check_c_source_compiles("__attribute__ ((weak)) int foo(int a) { return a*a; }
  int main(int argc, char** argv) {
  return foo(argc);}" LIBOMP_HAVE_WEAK_ATTRIBUTE)
set(CMAKE_REQUIRED_LIBRARIES psapi)
check_c_source_compiles("#include <windows.h>
  #include <psapi.h>
  int main(int artc, char** argv) {
    return EnumProcessModules(NULL, NULL, 0, NULL);
  }" LIBOMP_HAVE_PSAPI)
set(CMAKE_REQUIRED_LIBRARIES)
if(NOT LIBOMP_HAVE___BUILTIN_FRAME_ADDRESS)
  set(LIBOMP_HAVE_OMPT_SUPPORT FALSE)
else()
  if( # hardware architecture supported?
     ((LIBOMP_ARCH STREQUAL x86_64) OR
      (LIBOMP_ARCH STREQUAL i386) OR
#      (LIBOMP_ARCH STREQUAL arm) OR
      (LIBOMP_ARCH STREQUAL aarch64) OR
      (LIBOMP_ARCH STREQUAL aarch64_a64fx) OR
      (LIBOMP_ARCH STREQUAL ppc64le) OR
      (LIBOMP_ARCH STREQUAL ppc64) OR
      (LIBOMP_ARCH STREQUAL riscv64) OR
      (LIBOMP_ARCH STREQUAL loongarch64))
     AND # OS supported?
     ((WIN32 AND LIBOMP_HAVE_PSAPI) OR APPLE OR (NOT WIN32 AND LIBOMP_HAVE_WEAK_ATTRIBUTE)))
    set(LIBOMP_HAVE_OMPT_SUPPORT TRUE)
  else()
    set(LIBOMP_HAVE_OMPT_SUPPORT FALSE)
  endif()
endif()

set(LIBOMP_HAVE_OMPT_SUPPORT ${LIBOMP_HAVE_OMPT_SUPPORT} PARENT_SCOPE)

# Check if HWLOC support is available
if(${LIBOMP_USE_HWLOC})
  find_path(LIBOMP_HWLOC_INCLUDE_DIR NAMES hwloc.h HINTS ${LIBOMP_HWLOC_INSTALL_DIR} PATH_SUFFIXES include)
  set(CMAKE_REQUIRED_INCLUDES ${LIBOMP_HWLOC_INCLUDE_DIR})
  check_include_file(hwloc.h LIBOMP_HAVE_HWLOC_H)
  set(CMAKE_REQUIRED_INCLUDES)
  find_library(LIBOMP_HWLOC_LIBRARY
    NAMES hwloc libhwloc
    HINTS ${LIBOMP_HWLOC_INSTALL_DIR}/lib)
  if(LIBOMP_HWLOC_LIBRARY)
    check_library_exists(${LIBOMP_HWLOC_LIBRARY} hwloc_topology_init
      ${LIBOMP_HWLOC_INSTALL_DIR}/lib LIBOMP_HAVE_LIBHWLOC)
    get_filename_component(LIBOMP_HWLOC_LIBRARY_DIR ${LIBOMP_HWLOC_LIBRARY} PATH)
  endif()
  if(LIBOMP_HAVE_HWLOC_H AND LIBOMP_HAVE_LIBHWLOC AND LIBOMP_HWLOC_LIBRARY)
    set(LIBOMP_HAVE_HWLOC TRUE)
  else()
    set(LIBOMP_HAVE_HWLOC FALSE)
    libomp_say("Could not find hwloc")
  endif()
endif()

# Check if ThreadSanitizer support is available
if("${CMAKE_SYSTEM_NAME}" MATCHES "Linux" AND ${INTEL64})
  set(LIBOMP_HAVE_TSAN_SUPPORT TRUE)
else()
  set(LIBOMP_HAVE_TSAN_SUPPORT FALSE)
endif()
