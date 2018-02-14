#
#//===----------------------------------------------------------------------===//
#//
#//                     The LLVM Compiler Infrastructure
#//
#// This file is dual licensed under the MIT and the University of Illinois Open
#// Source Licenses. See LICENSE.txt for details.
#//
#//===----------------------------------------------------------------------===//
#

include(CheckCCompilerFlag)
include(CheckCSourceCompiles)
include(CheckCXXCompilerFlag)
include(CheckIncludeFile)
include(CheckLibraryExists)
include(CheckIncludeFiles)
include(LibompCheckLinkerFlag)
include(LibompCheckFortranFlag)

# Check for versioned symbols
function(libomp_check_version_symbols retval)
  set(source_code
    "#include <stdio.h>
    void func1() { printf(\"Hello\"); }
    void func2() { printf(\"World\"); }
    __asm__(\".symver func1, func@VER1\");
    __asm__(\".symver func2, func@VER2\");
    int main() {
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

# Checking C, CXX, Linker Flags
check_cxx_compiler_flag(-fno-exceptions LIBOMP_HAVE_FNO_EXCEPTIONS_FLAG)
check_cxx_compiler_flag(-fno-rtti LIBOMP_HAVE_FNO_RTTI_FLAG)
check_c_compiler_flag("-x c++" LIBOMP_HAVE_X_CPP_FLAG)
check_c_compiler_flag(-Wunused-function LIBOMP_HAVE_WNO_UNUSED_FUNCTION_FLAG)
check_c_compiler_flag(-Wunused-local-typedef LIBOMP_HAVE_WNO_UNUSED_LOCAL_TYPEDEF_FLAG)
check_c_compiler_flag(-Wunused-value LIBOMP_HAVE_WNO_UNUSED_VALUE_FLAG)
check_c_compiler_flag(-Wunused-variable LIBOMP_HAVE_WNO_UNUSED_VARIABLE_FLAG)
check_c_compiler_flag(-Wswitch LIBOMP_HAVE_WNO_SWITCH_FLAG)
check_c_compiler_flag(-Wcovered-switch-default LIBOMP_HAVE_WNO_COVERED_SWITCH_DEFAULT_FLAG)
check_c_compiler_flag(-Wdeprecated-register LIBOMP_HAVE_WNO_DEPRECATED_REGISTER_FLAG)
check_c_compiler_flag(-Wsign-compare LIBOMP_HAVE_WNO_SIGN_COMPARE_FLAG)
check_c_compiler_flag(-Wgnu-anonymous-struct LIBOMP_HAVE_WNO_GNU_ANONYMOUS_STRUCT_FLAG)
check_c_compiler_flag(-Wunknown-pragmas LIBOMP_HAVE_WNO_UNKNOWN_PRAGMAS_FLAG)
check_c_compiler_flag(-Wmissing-field-initializers LIBOMP_HAVE_WNO_MISSING_FIELD_INITIALIZERS_FLAG)
check_c_compiler_flag(-Wmissing-braces LIBOMP_HAVE_WNO_MISSING_BRACES_FLAG)
check_c_compiler_flag(-Wcomment LIBOMP_HAVE_WNO_COMMENT_FLAG)
check_c_compiler_flag(-Wself-assign LIBOMP_HAVE_WNO_SELF_ASSIGN_FLAG)
check_c_compiler_flag(-Wvla-extension LIBOMP_HAVE_WNO_VLA_EXTENSION_FLAG)
check_c_compiler_flag(-Wformat-pedantic LIBOMP_HAVE_WNO_FORMAT_PEDANTIC_FLAG)
check_c_compiler_flag(-msse2 LIBOMP_HAVE_MSSE2_FLAG)
check_c_compiler_flag(-ftls-model=initial-exec LIBOMP_HAVE_FTLS_MODEL_FLAG)
libomp_check_architecture_flag(-mmic LIBOMP_HAVE_MMIC_FLAG)
libomp_check_architecture_flag(-m32 LIBOMP_HAVE_M32_FLAG)
if(WIN32)
  # Check Windows MSVC style flags.
  check_c_compiler_flag(/TP LIBOMP_HAVE_TP_FLAG)
  check_cxx_compiler_flag(/EHsc LIBOMP_HAVE_EHSC_FLAG)
  check_cxx_compiler_flag(/GS LIBOMP_HAVE_GS_FLAG)
  check_cxx_compiler_flag(/Oy- LIBOMP_HAVE_Oy__FLAG)
  check_cxx_compiler_flag(/arch:SSE2 LIBOMP_HAVE_ARCH_SSE2_FLAG)
  check_cxx_compiler_flag(/Qsafeseh LIBOMP_HAVE_QSAFESEH_FLAG)
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
else()
  # It is difficult to create a dummy assembly file that compiles into an
  # exectuable for every architecture and then check the C compiler to
  # see if -x assembler-with-cpp exists and works, so we assume it does for non-Windows.
  set(LIBOMP_HAVE_X_ASSEMBLER_WITH_CPP_FLAG TRUE)
endif()
if(${LIBOMP_FORTRAN_MODULES})
  libomp_check_fortran_flag(-m32 LIBOMP_HAVE_M32_FORTRAN_FLAG)
endif()

# Check linker flags
if(WIN32)
  libomp_check_linker_flag(/SAFESEH LIBOMP_HAVE_SAFESEH_FLAG)
elseif(NOT APPLE)
  libomp_check_linker_flag(-Wl,-x LIBOMP_HAVE_X_FLAG)
  libomp_check_linker_flag(-Wl,--warn-shared-textrel LIBOMP_HAVE_WARN_SHARED_TEXTREL_FLAG)
  libomp_check_linker_flag(-Wl,--as-needed LIBOMP_HAVE_AS_NEEDED_FLAG)
  libomp_check_linker_flag("-Wl,--version-script=${LIBOMP_SRC_DIR}/exports_so.txt" LIBOMP_HAVE_VERSION_SCRIPT_FLAG)
  libomp_check_linker_flag(-static-libgcc LIBOMP_HAVE_STATIC_LIBGCC_FLAG)
  libomp_check_linker_flag(-Wl,-z,noexecstack LIBOMP_HAVE_Z_NOEXECSTACK_FLAG)
  libomp_check_linker_flag(-Wl,-fini=__kmp_internal_end_fini LIBOMP_HAVE_FINI_FLAG)
endif()

# Check Intel(R) C Compiler specific flags
if(CMAKE_C_COMPILER_ID STREQUAL "Intel")
  check_cxx_compiler_flag(/Qlong_double LIBOMP_HAVE_LONG_DOUBLE_FLAG)
  check_cxx_compiler_flag(/Qdiag-disable:177 LIBOMP_HAVE_DIAG_DISABLE_177_FLAG)
  check_cxx_compiler_flag(/Qinline-min-size=1 LIBOMP_HAVE_INLINE_MIN_SIZE_FLAG)
  check_cxx_compiler_flag(-Qoption,cpp,--extended_float_types LIBOMP_HAVE_EXTENDED_FLOAT_TYPES_FLAG)
  check_cxx_compiler_flag(-falign-stack=maintain-16-byte LIBOMP_HAVE_FALIGN_STACK_FLAG)
  check_cxx_compiler_flag("-opt-streaming-stores never" LIBOMP_HAVE_OPT_STREAMING_STORES_FLAG)
  libomp_check_linker_flag(-static-intel LIBOMP_HAVE_STATIC_INTEL_FLAG)
  libomp_check_linker_flag(-no-intel-extensions LIBOMP_HAVE_NO_INTEL_EXTENSIONS_FLAG)
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
libomp_check_version_symbols(LIBOMP_HAVE_VERSION_SYMBOLS)

# Check if quad precision types are available
if(CMAKE_C_COMPILER_ID STREQUAL "GNU")
  set(LIBOMP_HAVE_QUAD_PRECISION TRUE)
elseif(CMAKE_C_COMPILER_ID STREQUAL "Intel")
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
check_include_files("windows.h;psapi.h" LIBOMP_HAVE_PSAPI_H)
check_library_exists(psapi EnumProcessModules "" LIBOMP_HAVE_LIBPSAPI)
if(LIBOMP_HAVE_PSAPI_H AND LIBOMP_HAVE_LIBPSAPI)
  set(LIBOMP_HAVE_PSAPI TRUE)
endif()
if(NOT LIBOMP_HAVE___BUILTIN_FRAME_ADDRESS)
  set(LIBOMP_HAVE_OMPT_SUPPORT FALSE)
else()
  if( # hardware architecture supported?
     ((LIBOMP_ARCH STREQUAL x86_64) OR
      (LIBOMP_ARCH STREQUAL i386) OR
#      (LIBOMP_ARCH STREQUAL arm) OR
      (LIBOMP_ARCH STREQUAL aarch64) OR
      (LIBOMP_ARCH STREQUAL ppc64le) OR
      (LIBOMP_ARCH STREQUAL ppc64))
     AND # OS supported?
     ((WIN32 AND LIBOMP_HAVE_PSAPI) OR APPLE OR (NOT WIN32 AND LIBOMP_HAVE_WEAK_ATTRIBUTE)))
    set(LIBOMP_HAVE_OMPT_SUPPORT TRUE)
  else()
    set(LIBOMP_HAVE_OMPT_SUPPORT FALSE)
  endif()
endif()

# Check if HWLOC support is available
if(${LIBOMP_USE_HWLOC})
  set(CMAKE_REQUIRED_INCLUDES ${LIBOMP_HWLOC_INSTALL_DIR}/include)
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
