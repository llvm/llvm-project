# This CMake module is responsible for interpreting the user defined LLVM_
# options and executing the appropriate CMake commands to realize the users'
# selections.

# This is commonly needed so make sure it's defined before we include anything
# else.
string(TOUPPER "${CMAKE_BUILD_TYPE}" uppercase_CMAKE_BUILD_TYPE)

include(CheckCompilerVersion)
include(HandleLLVMStdlib)
include(CheckCCompilerFlag)
include(CheckCXXCompilerFlag)
include(CheckSymbolExists)
include(CMakeDependentOption)
include(LLVMProcessSources)

if(CMAKE_LINKER MATCHES ".*lld" OR (LLVM_USE_LINKER STREQUAL "lld" OR LLVM_ENABLE_LLD))
  set(LINKER_IS_LLD TRUE)
else()
  set(LINKER_IS_LLD FALSE)
endif()

if(CMAKE_LINKER MATCHES "lld-link" OR (MSVC AND (LLVM_USE_LINKER STREQUAL "lld" OR LLVM_ENABLE_LLD)))
  set(LINKER_IS_LLD_LINK TRUE)
else()
  set(LINKER_IS_LLD_LINK FALSE)
endif()

set(LLVM_ENABLE_LTO OFF CACHE STRING "Build LLVM with LTO. May be specified as Thin or Full to use a particular kind of LTO")
string(TOUPPER "${LLVM_ENABLE_LTO}" uppercase_LLVM_ENABLE_LTO)

# Ninja Job Pool support
# The following only works with the Ninja generator in CMake >= 3.0.
set(LLVM_PARALLEL_COMPILE_JOBS "" CACHE STRING
  "Define the maximum number of concurrent compilation jobs (Ninja only).")
if(LLVM_PARALLEL_COMPILE_JOBS)
  if(NOT CMAKE_GENERATOR STREQUAL "Ninja")
    message(WARNING "Job pooling is only available with Ninja generators.")
  else()
    set_property(GLOBAL APPEND PROPERTY JOB_POOLS compile_job_pool=${LLVM_PARALLEL_COMPILE_JOBS})
    set(CMAKE_JOB_POOL_COMPILE compile_job_pool)
  endif()
endif()

set(LLVM_PARALLEL_LINK_JOBS "" CACHE STRING
  "Define the maximum number of concurrent link jobs (Ninja only).")
if(CMAKE_GENERATOR STREQUAL "Ninja")
  if(NOT LLVM_PARALLEL_LINK_JOBS AND uppercase_LLVM_ENABLE_LTO STREQUAL "THIN")
    message(STATUS "ThinLTO provides its own parallel linking - limiting parallel link jobs to 2.")
    set(LLVM_PARALLEL_LINK_JOBS "2")
  endif()
  if(LLVM_PARALLEL_LINK_JOBS)
    set_property(GLOBAL APPEND PROPERTY JOB_POOLS link_job_pool=${LLVM_PARALLEL_LINK_JOBS})
    set(CMAKE_JOB_POOL_LINK link_job_pool)
  endif()
elseif(LLVM_PARALLEL_LINK_JOBS)
  message(WARNING "Job pooling is only available with Ninja generators.")
endif()

if( LLVM_ENABLE_ASSERTIONS )
  # MSVC doesn't like _DEBUG on release builds. See PR 4379.
  if( NOT MSVC )
    add_definitions( -D_DEBUG )
  endif()
  # On non-Debug builds cmake automatically defines NDEBUG, so we
  # explicitly undefine it:
  if( NOT uppercase_CMAKE_BUILD_TYPE STREQUAL "DEBUG" )
    # NOTE: use `add_compile_options` rather than `add_definitions` since
    # `add_definitions` does not support generator expressions.
    add_compile_options($<$<OR:$<COMPILE_LANGUAGE:C>,$<COMPILE_LANGUAGE:CXX>>:-UNDEBUG>)
    if (MSVC)
      # Also remove /D NDEBUG to avoid MSVC warnings about conflicting defines.
      foreach (flags_var_to_scrub
          CMAKE_CXX_FLAGS_RELEASE
          CMAKE_CXX_FLAGS_RELWITHDEBINFO
          CMAKE_CXX_FLAGS_MINSIZEREL
          CMAKE_C_FLAGS_RELEASE
          CMAKE_C_FLAGS_RELWITHDEBINFO
          CMAKE_C_FLAGS_MINSIZEREL)
        string (REGEX REPLACE "(^| )[/-]D *NDEBUG($| )" " "
          "${flags_var_to_scrub}" "${${flags_var_to_scrub}}")
      endforeach()
     endif()
  endif()
endif()

if(LLVM_ENABLE_EXPENSIVE_CHECKS)
  add_definitions(-DEXPENSIVE_CHECKS)

  # In some libstdc++ versions, std::min_element is not constexpr when
  # _GLIBCXX_DEBUG is enabled.
  CHECK_CXX_SOURCE_COMPILES("
    #define _GLIBCXX_DEBUG
    #include <algorithm>
    int main(int argc, char** argv) {
      static constexpr int data[] = {0, 1};
      constexpr const int* min_elt = std::min_element(&data[0], &data[2]);
      return 0;
    }" CXX_SUPPORTS_GLIBCXX_DEBUG)
  if(CXX_SUPPORTS_GLIBCXX_DEBUG)
    add_definitions(-D_GLIBCXX_DEBUG)
  else()
    add_definitions(-D_GLIBCXX_ASSERTIONS)
  endif()
endif()

if (LLVM_ENABLE_STRICT_FIXED_SIZE_VECTORS)
  add_definitions(-DSTRICT_FIXED_SIZE_VECTORS)
endif()

string(TOUPPER "${LLVM_ABI_BREAKING_CHECKS}" uppercase_LLVM_ABI_BREAKING_CHECKS)

if( uppercase_LLVM_ABI_BREAKING_CHECKS STREQUAL "WITH_ASSERTS" )
  if( LLVM_ENABLE_ASSERTIONS )
    set( LLVM_ENABLE_ABI_BREAKING_CHECKS 1 )
  endif()
elseif( uppercase_LLVM_ABI_BREAKING_CHECKS STREQUAL "FORCE_ON" )
  set( LLVM_ENABLE_ABI_BREAKING_CHECKS 1 )
elseif( uppercase_LLVM_ABI_BREAKING_CHECKS STREQUAL "FORCE_OFF" )
  # We don't need to do anything special to turn off ABI breaking checks.
elseif( NOT DEFINED LLVM_ABI_BREAKING_CHECKS )
  # Treat LLVM_ABI_BREAKING_CHECKS like "FORCE_OFF" when it has not been
  # defined.
else()
  message(FATAL_ERROR "Unknown value for LLVM_ABI_BREAKING_CHECKS: \"${LLVM_ABI_BREAKING_CHECKS}\"!")
endif()

if( LLVM_REVERSE_ITERATION )
  set( LLVM_ENABLE_REVERSE_ITERATION 1 )
endif()

if(WIN32)
  set(LLVM_HAVE_LINK_VERSION_SCRIPT 0)
  if(CYGWIN)
    set(LLVM_ON_WIN32 0)
    set(LLVM_ON_UNIX 1)
  else(CYGWIN)
    set(LLVM_ON_WIN32 1)
    set(LLVM_ON_UNIX 0)
  endif(CYGWIN)
else(WIN32)
  if(FUCHSIA OR UNIX)
    set(LLVM_ON_WIN32 0)
    set(LLVM_ON_UNIX 1)
    if(APPLE OR ${CMAKE_SYSTEM_NAME} MATCHES "AIX")
      set(LLVM_HAVE_LINK_VERSION_SCRIPT 0)
    else()
      set(LLVM_HAVE_LINK_VERSION_SCRIPT 1)
    endif()
  else(FUCHSIA OR UNIX)
    MESSAGE(SEND_ERROR "Unable to determine platform")
  endif(FUCHSIA OR UNIX)
endif(WIN32)

if (CMAKE_SYSTEM_NAME MATCHES "OS390")
  set(LLVM_HAVE_LINK_VERSION_SCRIPT 0)
endif()

set(EXEEXT ${CMAKE_EXECUTABLE_SUFFIX})
set(LTDL_SHLIB_EXT ${CMAKE_SHARED_LIBRARY_SUFFIX})

# We use *.dylib rather than *.so on darwin, but we stick with *.so on AIX.
if(${CMAKE_SYSTEM_NAME} MATCHES "AIX")
  set(LLVM_PLUGIN_EXT ${CMAKE_SHARED_MODULE_SUFFIX})
else()
  set(LLVM_PLUGIN_EXT ${CMAKE_SHARED_LIBRARY_SUFFIX})
endif()

if(APPLE)
  # Darwin-specific linker flags for loadable modules.
  set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -Wl,-flat_namespace -Wl,-undefined -Wl,suppress")
endif()

if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
  # RHEL7 has ar and ranlib being non-deterministic by default. The D flag forces determinism,
  # however only GNU version of ar and ranlib (2.27) have this option.
  # RHEL DTS7 is also affected by this, which uses GNU binutils 2.28
  execute_process(COMMAND ${CMAKE_AR} rD t.a
                  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                  RESULT_VARIABLE AR_RESULT
                  OUTPUT_QUIET
                  ERROR_QUIET
                  )
  if(${AR_RESULT} EQUAL 0)
    execute_process(COMMAND ${CMAKE_RANLIB} -D t.a
                    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                    RESULT_VARIABLE RANLIB_RESULT
                    OUTPUT_QUIET
                    ERROR_QUIET
                    )
    if(${RANLIB_RESULT} EQUAL 0)
      set(CMAKE_C_ARCHIVE_CREATE "<CMAKE_AR> Dqc <TARGET> <LINK_FLAGS> <OBJECTS>"
          CACHE STRING "archive create command")
      set(CMAKE_C_ARCHIVE_APPEND "<CMAKE_AR> Dq  <TARGET> <LINK_FLAGS> <OBJECTS>")
      set(CMAKE_C_ARCHIVE_FINISH "<CMAKE_RANLIB> -D <TARGET>" CACHE STRING "ranlib command")

      set(CMAKE_CXX_ARCHIVE_CREATE "<CMAKE_AR> Dqc <TARGET> <LINK_FLAGS> <OBJECTS>"
          CACHE STRING "archive create command")
      set(CMAKE_CXX_ARCHIVE_APPEND "<CMAKE_AR> Dq  <TARGET> <LINK_FLAGS> <OBJECTS>")
      set(CMAKE_CXX_ARCHIVE_FINISH "<CMAKE_RANLIB> -D <TARGET>" CACHE STRING "ranlib command")
    endif()
    file(REMOVE ${CMAKE_BINARY_DIR}/t.a)
  endif()
endif()

if(${CMAKE_SYSTEM_NAME} MATCHES "AIX")
  # -fPIC does not enable the large code model for GCC on AIX but does for XL.
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    append("-mcmodel=large" CMAKE_CXX_FLAGS CMAKE_C_FLAGS)
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "XL")
    # XL generates a small number of relocations not of the large model, -bbigtoc is needed.
    append("-Wl,-bbigtoc"
           CMAKE_EXE_LINKER_FLAGS CMAKE_MODULE_LINKER_FLAGS CMAKE_SHARED_LINKER_FLAGS)
    # The default behaviour on AIX processes dynamic initialization of non-local variables with
    # static storage duration even for archive members that are otherwise unreferenced.
    # Since `--whole-archive` is not used by the LLVM build to keep such initializations for Linux,
    # we can limit the processing for archive members to only those that are otherwise referenced.
    append("-bcdtors:mbr"
           CMAKE_EXE_LINKER_FLAGS CMAKE_MODULE_LINKER_FLAGS CMAKE_SHARED_LINKER_FLAGS)
  endif()
  if(BUILD_SHARED_LIBS)
    # See rpath handling in AddLLVM.cmake
    # FIXME: Remove this warning if this rpath is no longer hardcoded.
    message(WARNING "Build and install environment path info may be exposed; binaries will also be unrelocatable.")
  endif()
endif()

# Pass -Wl,-z,defs. This makes sure all symbols are defined. Otherwise a DSO
# build might work on ELF but fail on MachO/COFF.
if(NOT (CMAKE_SYSTEM_NAME MATCHES "Darwin|FreeBSD|OpenBSD|DragonFly|AIX|SunOS|OS390" OR
        WIN32 OR CYGWIN) AND
   NOT LLVM_USE_SANITIZER)
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-z,defs")
endif()

# Pass -Wl,-z,nodelete. This makes sure our shared libraries are not unloaded
# by dlclose(). We need that since the CLI API relies on cross-references
# between global objects which became horribly broken when one of the libraries
# is unloaded.
if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-z,nodelete")
  set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -Wl,-z,nodelete")
endif()


function(append value)
  foreach(variable ${ARGN})
    set(${variable} "${${variable}} ${value}" PARENT_SCOPE)
  endforeach(variable)
endfunction()

function(append_if condition value)
  if (${condition})
    foreach(variable ${ARGN})
      set(${variable} "${${variable}} ${value}" PARENT_SCOPE)
    endforeach(variable)
  endif()
endfunction()

macro(add_flag_if_supported flag name)
  check_c_compiler_flag("-Werror ${flag}" "C_SUPPORTS_${name}")
  append_if("C_SUPPORTS_${name}" "${flag}" CMAKE_C_FLAGS)
  check_cxx_compiler_flag("-Werror ${flag}" "CXX_SUPPORTS_${name}")
  append_if("CXX_SUPPORTS_${name}" "${flag}" CMAKE_CXX_FLAGS)
endmacro()

function(add_flag_or_print_warning flag name)
  check_c_compiler_flag("-Werror ${flag}" "C_SUPPORTS_${name}")
  check_cxx_compiler_flag("-Werror ${flag}" "CXX_SUPPORTS_${name}")
  if (C_SUPPORTS_${name} AND CXX_SUPPORTS_${name})
    message(STATUS "Building with ${flag}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${flag}" PARENT_SCOPE)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${flag}" PARENT_SCOPE)
    set(CMAKE_ASM_FLAGS "${CMAKE_ASM_FLAGS} ${flag}" PARENT_SCOPE)
  else()
    message(WARNING "${flag} is not supported.")
  endif()
endfunction()

if( LLVM_ENABLE_LLD )
  if ( LLVM_USE_LINKER )
    message(FATAL_ERROR "LLVM_ENABLE_LLD and LLVM_USE_LINKER can't be set at the same time")
  endif()
  # In case of MSVC cmake always invokes the linker directly, so the linker
  # should be specified by CMAKE_LINKER cmake variable instead of by -fuse-ld
  # compiler option.
  if ( NOT MSVC )
    set(LLVM_USE_LINKER "lld")
  endif()
endif()

if( LLVM_USE_LINKER )
  set(OLD_CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS})
  set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} -fuse-ld=${LLVM_USE_LINKER}")
  check_cxx_source_compiles("int main() { return 0; }" CXX_SUPPORTS_CUSTOM_LINKER)
  if ( NOT CXX_SUPPORTS_CUSTOM_LINKER )
    message(FATAL_ERROR "Host compiler does not support '-fuse-ld=${LLVM_USE_LINKER}'")
  endif()
  set(CMAKE_REQUIRED_FLAGS ${OLD_CMAKE_REQUIRED_FLAGS})
  append("-fuse-ld=${LLVM_USE_LINKER}"
    CMAKE_EXE_LINKER_FLAGS CMAKE_MODULE_LINKER_FLAGS CMAKE_SHARED_LINKER_FLAGS)
endif()

if( LLVM_ENABLE_PIC )
  if( XCODE )
    # Xcode has -mdynamic-no-pic on by default, which overrides -fPIC. I don't
    # know how to disable this, so just force ENABLE_PIC off for now.
    message(WARNING "-fPIC not supported with Xcode.")
  elseif( WIN32 OR CYGWIN)
    # On Windows all code is PIC. MinGW warns if -fPIC is used.
  else()
    add_flag_or_print_warning("-fPIC" FPIC)
    # Enable interprocedural optimizations for non-inline functions which would
    # otherwise be disabled due to GCC -fPIC's default.
    # Note: GCC<10.3 has a bug on SystemZ.
    #
    # Note: Clang allows IPO for -fPIC so this optimization is less effective.
    # Clang 13 has a bug related to -fsanitize-coverage
    # -fno-semantic-interposition (https://reviews.llvm.org/D117183).
    if ((CMAKE_COMPILER_IS_GNUCXX AND
         NOT (LLVM_NATIVE_ARCH STREQUAL "SystemZ" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 10.3))
       OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_COMPILER_VERSION GREATER_EQUAL 14))
      add_flag_if_supported("-fno-semantic-interposition" FNO_SEMANTIC_INTERPOSITION)
    endif()
  endif()
  # GCC for MIPS can miscompile LLVM due to PR37701.
  if(CMAKE_COMPILER_IS_GNUCXX AND LLVM_NATIVE_ARCH STREQUAL "Mips" AND
         NOT Uppercase_CMAKE_BUILD_TYPE STREQUAL "DEBUG")
    add_flag_or_print_warning("-fno-shrink-wrap" FNO_SHRINK_WRAP)
  endif()
  # gcc with -O3 -fPIC generates TLS sequences that violate the spec on
  # Solaris/sparcv9, causing executables created with the system linker
  # to SEGV (GCC PR target/96607).
  # clang with -O3 -fPIC generates code that SEGVs.
  # Both can be worked around by compiling with -O instead.
  if(${CMAKE_SYSTEM_NAME} STREQUAL "SunOS" AND LLVM_NATIVE_ARCH STREQUAL "Sparc")
    llvm_replace_compiler_option(CMAKE_CXX_FLAGS_RELEASE "-O[23]" "-O")
    llvm_replace_compiler_option(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O[23]" "-O")
  endif()
endif()

if(NOT WIN32 AND NOT CYGWIN AND NOT (${CMAKE_SYSTEM_NAME} MATCHES "AIX"))
  # MinGW warns if -fvisibility-inlines-hidden is used.
  # GCC on AIX warns if -fvisibility-inlines-hidden is used and Clang on AIX doesn't currently support visibility.
  check_cxx_compiler_flag("-fvisibility-inlines-hidden" SUPPORTS_FVISIBILITY_INLINES_HIDDEN_FLAG)
  append_if(SUPPORTS_FVISIBILITY_INLINES_HIDDEN_FLAG "-fvisibility-inlines-hidden" CMAKE_CXX_FLAGS)
endif()

if(CMAKE_SIZEOF_VOID_P EQUAL 8 AND MINGW)
  add_definitions( -D_FILE_OFFSET_BITS=64 )
endif()

if( CMAKE_SIZEOF_VOID_P EQUAL 8 AND NOT WIN32 )
  # TODO: support other platforms and toolchains.
  if( LLVM_BUILD_32_BITS )
    message(STATUS "Building 32 bits executables and libraries.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m32")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -m32")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -m32")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -m32")
    set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -m32")

    # FIXME: CMAKE_SIZEOF_VOID_P is still 8
    add_definitions(-D_LARGEFILE_SOURCE)
    add_definitions(-D_FILE_OFFSET_BITS=64)
  endif( LLVM_BUILD_32_BITS )
endif( CMAKE_SIZEOF_VOID_P EQUAL 8 AND NOT WIN32 )

# If building on a GNU specific 32-bit system, make sure off_t is 64 bits
# so that off_t can stored offset > 2GB.
# Android until version N (API 24) doesn't support it.
if (ANDROID AND (ANDROID_NATIVE_API_LEVEL LESS 24))
  set(LLVM_FORCE_SMALLFILE_FOR_ANDROID TRUE)
endif()
if( CMAKE_SIZEOF_VOID_P EQUAL 4 AND NOT LLVM_FORCE_SMALLFILE_FOR_ANDROID)
  # FIXME: It isn't handled in LLVM_BUILD_32_BITS.
  add_definitions( -D_LARGEFILE_SOURCE )
  add_definitions( -D_FILE_OFFSET_BITS=64 )
endif()

if( XCODE )
  # For Xcode enable several build settings that correspond to
  # many warnings that are on by default in Clang but are
  # not enabled for historical reasons.  For versions of Xcode
  # that do not support these options they will simply
  # be ignored.
  set(CMAKE_XCODE_ATTRIBUTE_GCC_WARN_ABOUT_RETURN_TYPE "YES")
  set(CMAKE_XCODE_ATTRIBUTE_GCC_WARN_ABOUT_MISSING_NEWLINE "YES")
  set(CMAKE_XCODE_ATTRIBUTE_GCC_WARN_UNUSED_VALUE "YES")
  set(CMAKE_XCODE_ATTRIBUTE_GCC_WARN_UNUSED_VARIABLE "YES")
  set(CMAKE_XCODE_ATTRIBUTE_GCC_WARN_SIGN_COMPARE "YES")
  set(CMAKE_XCODE_ATTRIBUTE_GCC_WARN_UNUSED_FUNCTION "YES")
  set(CMAKE_XCODE_ATTRIBUTE_GCC_WARN_INITIALIZER_NOT_FULLY_BRACKETED "YES")
  set(CMAKE_XCODE_ATTRIBUTE_GCC_WARN_HIDDEN_VIRTUAL_FUNCTIONS "YES")
  set(CMAKE_XCODE_ATTRIBUTE_GCC_WARN_UNINITIALIZED_AUTOS "YES")
  set(CMAKE_XCODE_ATTRIBUTE_CLANG_WARN_BOOL_CONVERSION "YES")
  set(CMAKE_XCODE_ATTRIBUTE_CLANG_WARN_EMPTY_BODY "YES")
  set(CMAKE_XCODE_ATTRIBUTE_CLANG_WARN_ENUM_CONVERSION "YES")
  set(CMAKE_XCODE_ATTRIBUTE_CLANG_WARN_INT_CONVERSION "YES")
  set(CMAKE_XCODE_ATTRIBUTE_CLANG_WARN_CONSTANT_CONVERSION "YES")
  set(CMAKE_XCODE_ATTRIBUTE_GCC_WARN_NON_VIRTUAL_DESTRUCTOR "YES")
endif()

# On Win32 using MS tools, provide an option to set the number of parallel jobs
# to use.
if( MSVC_IDE )
  set(LLVM_COMPILER_JOBS "0" CACHE STRING
    "Number of parallel compiler jobs. 0 means use all processors. Default is 0.")
  if( NOT LLVM_COMPILER_JOBS STREQUAL "1" )
    if( LLVM_COMPILER_JOBS STREQUAL "0" )
      add_definitions( /MP )
    else()
      message(STATUS "Number of parallel compiler jobs set to " ${LLVM_COMPILER_JOBS})
      add_definitions( /MP${LLVM_COMPILER_JOBS} )
    endif()
  else()
    message(STATUS "Parallel compilation disabled")
  endif()
endif()

# set stack reserved size to ~10MB
if(MSVC)
  # CMake previously automatically set this value for MSVC builds, but the
  # behavior was changed in CMake 2.8.11 (Issue 12437) to use the MSVC default
  # value (1 MB) which is not enough for us in tasks such as parsing recursive
  # C++ templates in Clang.
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /STACK:10000000")
elseif(MINGW) # FIXME: Also cygwin?
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--stack,16777216")

  # Pass -mbig-obj to mingw gas to avoid COFF 2**16 section limit.
  if (NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    append("-Wa,-mbig-obj" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
  endif()
endif()

option(LLVM_ENABLE_WARNINGS "Enable compiler warnings." ON)

if( MSVC )
  include(ChooseMSVCCRT)

  # Add definitions that make MSVC much less annoying.
  add_definitions(
    # For some reason MS wants to deprecate a bunch of standard functions...
    -D_CRT_SECURE_NO_DEPRECATE
    -D_CRT_SECURE_NO_WARNINGS
    -D_CRT_NONSTDC_NO_DEPRECATE
    -D_CRT_NONSTDC_NO_WARNINGS
    -D_SCL_SECURE_NO_DEPRECATE
    -D_SCL_SECURE_NO_WARNINGS
    )

  # Tell MSVC to use the Unicode version of the Win32 APIs instead of ANSI.
  add_definitions(
    -DUNICODE
    -D_UNICODE
  )

  if (LLVM_WINSYSROOT)
    if (NOT CLANG_CL)
      message(ERROR "LLVM_WINSYSROOT requires clang-cl")
    endif()
    append("/winsysroot${LLVM_WINSYSROOT}" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
  endif()

  if (LLVM_ENABLE_WERROR)
    append("/WX" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
  endif (LLVM_ENABLE_WERROR)

  append("/Zc:inline" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)

  # Some projects use the __cplusplus preprocessor macro to check support for
  # a particular version of the C++ standard. When this option is not specified
  # explicitly, macro's value is "199711L" that implies C++98 Standard.
  # https://devblogs.microsoft.com/cppblog/msvc-now-correctly-reports-__cplusplus/
  append("/Zc:__cplusplus" CMAKE_CXX_FLAGS)

  # Allow users to request PDBs in release mode. CMake offeres the
  # RelWithDebInfo configuration, but it uses different optimization settings
  # (/Ob1 vs /Ob2 or -O2 vs -O3). LLVM provides this flag so that users can get
  # PDBs without changing codegen.
  option(LLVM_ENABLE_PDB OFF)
  if (LLVM_ENABLE_PDB AND uppercase_CMAKE_BUILD_TYPE STREQUAL "RELEASE")
    append("/Zi" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
    # /DEBUG disables linker GC and ICF, but we want those in Release mode.
    append("/DEBUG /OPT:REF /OPT:ICF"
          CMAKE_EXE_LINKER_FLAGS CMAKE_MODULE_LINKER_FLAGS
          CMAKE_SHARED_LINKER_FLAGS)
  endif()

  # Get all linker flags in upper case form so we can search them.
  string(CONCAT all_linker_flags_uppercase
     ${CMAKE_EXE_LINKER_FLAGS_${uppercase_CMAKE_BUILD_TYPE}} " "
     ${CMAKE_EXE_LINKER_FLAGS} " "
     ${CMAKE_MODULE_LINKER_FLAGS_${uppercase_CMAKE_BUILD_TYPE}} " "
     ${CMAKE_MODULE_LINKER_FLAGS} " "
     ${CMAKE_SHARED_LINKER_FLAGS_${uppercase_CMAKE_BUILD_TYPE}} " "
     ${CMAKE_SHARED_LINKER_FLAGS})
  string(TOUPPER "${all_linker_flags_uppercase}" all_linker_flags_uppercase)

  if (CLANG_CL AND LINKER_IS_LLD)
    # If we are using clang-cl with lld-link and /debug is present in any of the
    # linker flag variables, pass -gcodeview-ghash to the compiler to speed up
    # linking. This flag is orthogonal from /Zi, /Z7, and other flags that
    # enable debug info emission, and only has an effect if those are also in
    # use.
    string(FIND "${all_linker_flags_uppercase}" "/DEBUG" linker_flag_idx)
    if (${linker_flag_idx} GREATER -1)
      add_flag_if_supported("-gcodeview-ghash" GCODEVIEW_GHASH)
    endif()
  endif()

  # Disable string literal const->non-const type conversion.
  # "When specified, the compiler requires strict const-qualification
  # conformance for pointers initialized by using string literals."
  append("/Zc:strictStrings" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)

  # "Generate Intrinsic Functions".
  append("/Oi" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)

  # "Enforce type conversion rules".
  append("/Zc:rvalueCast" CMAKE_CXX_FLAGS)

  if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND NOT LLVM_ENABLE_LTO)
    # clang-cl and cl by default produce non-deterministic binaries because
    # link.exe /incremental requires a timestamp in the .obj file.  clang-cl
    # has the flag /Brepro to force deterministic binaries. We want to pass that
    # whenever you're building with clang unless you're passing /incremental
    # or using LTO (/Brepro with LTO would result in a warning about the flag
    # being unused, because we're not generating object files).
    # This checks CMAKE_CXX_COMPILER_ID in addition to check_cxx_compiler_flag()
    # because cl.exe does not emit an error on flags it doesn't understand,
    # letting check_cxx_compiler_flag() claim it understands all flags.
    check_cxx_compiler_flag("/Brepro" SUPPORTS_BREPRO)
    if (SUPPORTS_BREPRO)
      # Check if /INCREMENTAL is passed to the linker and complain that it
      # won't work with /Brepro.
      string(FIND "${all_linker_flags_uppercase}" "/INCREMENTAL" linker_flag_idx)
      if (${linker_flag_idx} GREATER -1)
        message(WARNING "/Brepro not compatible with /INCREMENTAL linking - builds will be non-deterministic")
      else()
        append("/Brepro" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
      endif()
    endif()
  endif()
  # By default MSVC has a 2^16 limit on the number of sections in an object file,
  # but in many objects files need more than that. This flag is to increase the
  # number of sections.
  append("/bigobj" CMAKE_CXX_FLAGS)
endif( MSVC )

# Warnings-as-errors handling for GCC-compatible compilers:
if ( LLVM_COMPILER_IS_GCC_COMPATIBLE )
  append_if(LLVM_ENABLE_WERROR "-Werror" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
  append_if(LLVM_ENABLE_WERROR "-Wno-error" CMAKE_REQUIRED_FLAGS)
endif( LLVM_COMPILER_IS_GCC_COMPATIBLE )

# Specific default warnings-as-errors for compilers accepting GCC-compatible warning flags:
if ( LLVM_COMPILER_IS_GCC_COMPATIBLE OR CMAKE_CXX_COMPILER_ID MATCHES "XL" )
  add_flag_if_supported("-Werror=date-time" WERROR_DATE_TIME)
  add_flag_if_supported("-Werror=unguarded-availability-new" WERROR_UNGUARDED_AVAILABILITY_NEW)
endif( LLVM_COMPILER_IS_GCC_COMPATIBLE OR CMAKE_CXX_COMPILER_ID MATCHES "XL" )

# Modules enablement for GCC-compatible compilers:
if ( LLVM_COMPILER_IS_GCC_COMPATIBLE AND LLVM_ENABLE_MODULES )
  set(OLD_CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS})
  set(module_flags "-fmodules -fmodules-cache-path=${PROJECT_BINARY_DIR}/module.cache")
  if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    # On Darwin -fmodules does not imply -fcxx-modules.
    set(module_flags "${module_flags} -fcxx-modules")
  endif()
  if (LLVM_ENABLE_LOCAL_SUBMODULE_VISIBILITY)
    set(module_flags "${module_flags} -Xclang -fmodules-local-submodule-visibility")
  endif()
  if (LLVM_ENABLE_MODULE_DEBUGGING AND
      ((uppercase_CMAKE_BUILD_TYPE STREQUAL "DEBUG") OR
       (uppercase_CMAKE_BUILD_TYPE STREQUAL "RELWITHDEBINFO")))
    set(module_flags "${module_flags} -gmodules")
  endif()
  set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} ${module_flags}")

  # Check that we can build code with modules enabled, and that repeatedly
  # including <cassert> still manages to respect NDEBUG properly.
  CHECK_CXX_SOURCE_COMPILES("#undef NDEBUG
                             #include <cassert>
                             #define NDEBUG
                             #include <cassert>
                             int main() { assert(this code is not compiled); }"
                             CXX_SUPPORTS_MODULES)
  set(CMAKE_REQUIRED_FLAGS ${OLD_CMAKE_REQUIRED_FLAGS})
  if (CXX_SUPPORTS_MODULES)
    append("${module_flags}" CMAKE_CXX_FLAGS)
  else()
    message(FATAL_ERROR "LLVM_ENABLE_MODULES is not supported by this compiler")
  endif()
endif( LLVM_COMPILER_IS_GCC_COMPATIBLE AND LLVM_ENABLE_MODULES )

if (MSVC)
  if (NOT CLANG_CL)
    set(msvc_warning_flags
      # Disabled warnings.
      -wd4141 # Suppress ''modifier' : used more than once' (because of __forceinline combined with inline)
      -wd4146 # Suppress 'unary minus operator applied to unsigned type, result still unsigned'
      -wd4244 # Suppress ''argument' : conversion from 'type1' to 'type2', possible loss of data'
      -wd4267 # Suppress ''var' : conversion from 'size_t' to 'type', possible loss of data'
      -wd4291 # Suppress ''declaration' : no matching operator delete found; memory will not be freed if initialization throws an exception'
      -wd4351 # Suppress 'new behavior: elements of array 'array' will be default initialized'
      -wd4456 # Suppress 'declaration of 'var' hides local variable'
      -wd4457 # Suppress 'declaration of 'var' hides function parameter'
      -wd4458 # Suppress 'declaration of 'var' hides class member'
      -wd4459 # Suppress 'declaration of 'var' hides global declaration'
      -wd4503 # Suppress ''identifier' : decorated name length exceeded, name was truncated'
      -wd4624 # Suppress ''derived class' : destructor could not be generated because a base class destructor is inaccessible'
      -wd4722 # Suppress 'function' : destructor never returns, potential memory leak
      -wd4100 # Suppress 'unreferenced formal parameter'
      -wd4127 # Suppress 'conditional expression is constant'
      -wd4512 # Suppress 'assignment operator could not be generated'
      -wd4505 # Suppress 'unreferenced local function has been removed'
      -wd4610 # Suppress '<class> can never be instantiated'
      -wd4510 # Suppress 'default constructor could not be generated'
      -wd4702 # Suppress 'unreachable code'
      -wd4245 # Suppress ''conversion' : conversion from 'type1' to 'type2', signed/unsigned mismatch'
      -wd4706 # Suppress 'assignment within conditional expression'
      -wd4310 # Suppress 'cast truncates constant value'
      -wd4701 # Suppress 'potentially uninitialized local variable'
      -wd4703 # Suppress 'potentially uninitialized local pointer variable'
      -wd4389 # Suppress 'signed/unsigned mismatch'
      -wd4611 # Suppress 'interaction between '_setjmp' and C++ object destruction is non-portable'
      -wd4805 # Suppress 'unsafe mix of type <type> and type <type> in operation'
      -wd4204 # Suppress 'nonstandard extension used : non-constant aggregate initializer'
      -wd4577 # Suppress 'noexcept used with no exception handling mode specified; termination on exception is not guaranteed'
      -wd4091 # Suppress 'typedef: ignored on left of '' when no variable is declared'
          # C4592 is disabled because of false positives in Visual Studio 2015
          # Update 1. Re-evaluate the usefulness of this diagnostic with Update 2.
      -wd4592 # Suppress ''var': symbol will be dynamically initialized (implementation limitation)
      -wd4319 # Suppress ''operator' : zero extending 'type' to 'type' of greater size'
          # C4709 is disabled because of a bug with Visual Studio 2017 as of
          # v15.8.8. Re-evaluate the usefulness of this diagnostic when the bug
          # is fixed.
      -wd4709 # Suppress comma operator within array index expression

      # Ideally, we'd like this warning to be enabled, but even MSVC 2019 doesn't
      # support the 'aligned' attribute in the way that clang sources requires (for
      # any code that uses the LLVM_ALIGNAS macro), so this is must be disabled to
      # avoid unwanted alignment warnings.
      -wd4324 # Suppress 'structure was padded due to __declspec(align())'

      # Promoted warnings.
      -w14062 # Promote 'enumerator in switch of enum is not handled' to level 1 warning.

      # Promoted warnings to errors.
      -we4238 # Promote 'nonstandard extension used : class rvalue used as lvalue' to error.
      )
  endif(NOT CLANG_CL)

  # Enable warnings
  if (LLVM_ENABLE_WARNINGS)
    # Put /W4 in front of all the -we flags. cl.exe doesn't care, but for
    # clang-cl having /W4 after the -we flags will re-enable the warnings
    # disabled by -we.
    set(msvc_warning_flags "/W4 ${msvc_warning_flags}")
    # CMake appends /W3 by default, and having /W3 followed by /W4 will result in
    # cl : Command line warning D9025 : overriding '/W3' with '/W4'.  Since this is
    # a command line warning and not a compiler warning, it cannot be suppressed except
    # by fixing the command line.
    string(REGEX REPLACE " /W[0-4]" "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
    string(REGEX REPLACE " /W[0-4]" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

    if (LLVM_ENABLE_PEDANTIC)
      # No MSVC equivalent available
    endif (LLVM_ENABLE_PEDANTIC)
  endif (LLVM_ENABLE_WARNINGS)

  foreach(flag ${msvc_warning_flags})
    append("${flag}" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
  endforeach(flag)
endif (MSVC)

if (LLVM_ENABLE_WARNINGS AND (LLVM_COMPILER_IS_GCC_COMPATIBLE OR CLANG_CL))

  # Don't add -Wall for clang-cl, because it maps -Wall to -Weverything for
  # MSVC compatibility.  /W4 is added above instead.
  if (NOT CLANG_CL)
    append("-Wall" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
  endif()

  append("-Wextra -Wno-unused-parameter -Wwrite-strings" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
  append("-Wcast-qual" CMAKE_CXX_FLAGS)

  # Turn off missing field initializer warnings for gcc to avoid noise from
  # false positives with empty {}. Turn them on otherwise (they're off by
  # default for clang).
  check_cxx_compiler_flag("-Wmissing-field-initializers" CXX_SUPPORTS_MISSING_FIELD_INITIALIZERS_FLAG)
  if (CXX_SUPPORTS_MISSING_FIELD_INITIALIZERS_FLAG)
    if (CMAKE_COMPILER_IS_GNUCXX)
      append("-Wno-missing-field-initializers" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
    else()
      append("-Wmissing-field-initializers" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
    endif()
  endif()

  if (LLVM_ENABLE_PEDANTIC AND LLVM_COMPILER_IS_GCC_COMPATIBLE)
    append("-pedantic" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
    append("-Wno-long-long" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)

    # GCC warns about redundant toplevel semicolons (enabled by -pedantic
    # above), while Clang doesn't. Enable the corresponding Clang option to
    # pick up on these even in builds with Clang.
    add_flag_if_supported("-Wc++98-compat-extra-semi" CXX98_COMPAT_EXTRA_SEMI_FLAG)
  endif()

  add_flag_if_supported("-Wimplicit-fallthrough" IMPLICIT_FALLTHROUGH_FLAG)
  add_flag_if_supported("-Wcovered-switch-default" COVERED_SWITCH_DEFAULT_FLAG)
  append_if(USE_NO_UNINITIALIZED "-Wno-uninitialized" CMAKE_CXX_FLAGS)
  append_if(USE_NO_MAYBE_UNINITIALIZED "-Wno-maybe-uninitialized" CMAKE_CXX_FLAGS)

  # Disable -Wclass-memaccess, a C++-only warning from GCC 8 that fires on
  # LLVM's ADT classes.
  check_cxx_compiler_flag("-Wclass-memaccess" CXX_SUPPORTS_CLASS_MEMACCESS_FLAG)
  append_if(CXX_SUPPORTS_CLASS_MEMACCESS_FLAG "-Wno-class-memaccess" CMAKE_CXX_FLAGS)

  # Disable -Wredundant-move and -Wpessimizing-move on GCC>=9. GCC wants to
  # remove std::move in code like "A foo(ConvertibleToA a) {
  # return std::move(a); }", but this code does not compile (or uses the copy
  # constructor instead) on clang<=3.8. Clang also has a -Wredundant-move and
  # -Wpessimizing-move, but they only fire when the types match exactly, so we
  # can keep them here.
  if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    check_cxx_compiler_flag("-Wredundant-move" CXX_SUPPORTS_REDUNDANT_MOVE_FLAG)
    append_if(CXX_SUPPORTS_REDUNDANT_MOVE_FLAG "-Wno-redundant-move" CMAKE_CXX_FLAGS)
    check_cxx_compiler_flag("-Wpessimizing-move" CXX_SUPPORTS_PESSIMIZING_MOVE_FLAG)
    append_if(CXX_SUPPORTS_PESSIMIZING_MOVE_FLAG "-Wno-pessimizing-move" CMAKE_CXX_FLAGS)
  endif()

  # The LLVM libraries have no stable C++ API, so -Wnoexcept-type is not useful.
  check_cxx_compiler_flag("-Wnoexcept-type" CXX_SUPPORTS_NOEXCEPT_TYPE_FLAG)
  append_if(CXX_SUPPORTS_NOEXCEPT_TYPE_FLAG "-Wno-noexcept-type" CMAKE_CXX_FLAGS)

  # Check if -Wnon-virtual-dtor warns for a class marked final, when it has a
  # friend declaration. If it does, don't add -Wnon-virtual-dtor. The case is
  # considered unhelpful (https://gcc.gnu.org/PR102168).
  set(OLD_CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS})
  set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} -Werror=non-virtual-dtor")
  CHECK_CXX_SOURCE_COMPILES("class f {};
                             class base {friend f; public: virtual void anchor();protected: ~base();};
                             int main() { return 0; }"
                            CXX_WONT_WARN_ON_FINAL_NONVIRTUALDTOR)
  set(CMAKE_REQUIRED_FLAGS ${OLD_CMAKE_REQUIRED_FLAGS})
  append_if(CXX_WONT_WARN_ON_FINAL_NONVIRTUALDTOR "-Wnon-virtual-dtor" CMAKE_CXX_FLAGS)

  append("-Wdelete-non-virtual-dtor" CMAKE_CXX_FLAGS)

  # Enable -Wsuggest-override if it's available, and only if it doesn't
  # suggest adding 'override' to functions that are already marked 'final'
  # (which means it is disabled for GCC < 9.2).
  check_cxx_compiler_flag("-Wsuggest-override" CXX_SUPPORTS_SUGGEST_OVERRIDE_FLAG)
  if (CXX_SUPPORTS_SUGGEST_OVERRIDE_FLAG)
    set(OLD_CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS})
    set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} -Werror=suggest-override")
    CHECK_CXX_SOURCE_COMPILES("class base {public: virtual void anchor();};
                               class derived : base {public: void anchor() final;};
                               int main() { return 0; }"
                              CXX_WSUGGEST_OVERRIDE_ALLOWS_ONLY_FINAL)
    set(CMAKE_REQUIRED_FLAGS ${OLD_CMAKE_REQUIRED_FLAGS})
    append_if(CXX_WSUGGEST_OVERRIDE_ALLOWS_ONLY_FINAL "-Wsuggest-override" CMAKE_CXX_FLAGS)
  endif()

  # Check if -Wcomment is OK with an // comment ending with '\' if the next
  # line is also a // comment.
  set(OLD_CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS})
  set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} -Werror -Wcomment")
  CHECK_C_SOURCE_COMPILES("// \\\\\\n//\\nint main() {return 0;}"
                          C_WCOMMENT_ALLOWS_LINE_WRAP)
  set(CMAKE_REQUIRED_FLAGS ${OLD_CMAKE_REQUIRED_FLAGS})
  if (NOT C_WCOMMENT_ALLOWS_LINE_WRAP)
    append("-Wno-comment" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
  endif()

  # Enable -Wstring-conversion to catch misuse of string literals.
  add_flag_if_supported("-Wstring-conversion" STRING_CONVERSION_FLAG)

  # Prevent bugs that can happen with llvm's brace style.
  add_flag_if_supported("-Wmisleading-indentation" MISLEADING_INDENTATION_FLAG)
endif (LLVM_ENABLE_WARNINGS AND (LLVM_COMPILER_IS_GCC_COMPATIBLE OR CLANG_CL))

if (LLVM_COMPILER_IS_GCC_COMPATIBLE AND NOT LLVM_ENABLE_WARNINGS)
  append("-w" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
endif()

macro(append_common_sanitizer_flags)
  if (NOT MSVC)
    # Append -fno-omit-frame-pointer and turn on debug info to get better
    # stack traces.
    add_flag_if_supported("-fno-omit-frame-pointer" FNO_OMIT_FRAME_POINTER)
    if (NOT uppercase_CMAKE_BUILD_TYPE STREQUAL "DEBUG" AND
        NOT uppercase_CMAKE_BUILD_TYPE STREQUAL "RELWITHDEBINFO")
      add_flag_if_supported("-gline-tables-only" GLINE_TABLES_ONLY)
    endif()
    # Use -O1 even in debug mode, otherwise sanitizers slowdown is too large.
    if (uppercase_CMAKE_BUILD_TYPE STREQUAL "DEBUG" AND LLVM_OPTIMIZE_SANITIZED_BUILDS)
      add_flag_if_supported("-O1" O1)
    endif()
  elseif (CLANG_CL)
    # Keep frame pointers around.
    append("/Oy-" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
    # Always ask the linker to produce symbols with asan.
    append("/Z7" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
    append("-debug" CMAKE_EXE_LINKER_FLAGS CMAKE_MODULE_LINKER_FLAGS CMAKE_SHARED_LINKER_FLAGS)
  endif()
endmacro()

# Turn on sanitizers if necessary.
if(LLVM_USE_SANITIZER)
  if (LLVM_ON_UNIX)
    if (LLVM_USE_SANITIZER STREQUAL "Address")
      append_common_sanitizer_flags()
      append("-fsanitize=address" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
    elseif (LLVM_USE_SANITIZER STREQUAL "HWAddress")
      append_common_sanitizer_flags()
      append("-fsanitize=hwaddress" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
    elseif (LLVM_USE_SANITIZER MATCHES "Memory(WithOrigins)?")
      append_common_sanitizer_flags()
      append("-fsanitize=memory" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
      if(LLVM_USE_SANITIZER STREQUAL "MemoryWithOrigins")
        append("-fsanitize-memory-track-origins" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
      endif()
    elseif (LLVM_USE_SANITIZER STREQUAL "Undefined")
      append_common_sanitizer_flags()
      append("${LLVM_UBSAN_FLAGS}" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
    elseif (LLVM_USE_SANITIZER STREQUAL "Thread")
      append_common_sanitizer_flags()
      append("-fsanitize=thread" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
    elseif (LLVM_USE_SANITIZER STREQUAL "DataFlow")
      append("-fsanitize=dataflow" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
    elseif (LLVM_USE_SANITIZER STREQUAL "Address;Undefined" OR
            LLVM_USE_SANITIZER STREQUAL "Undefined;Address")
      append_common_sanitizer_flags()
      append("-fsanitize=address" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
      append("${LLVM_UBSAN_FLAGS}" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
    elseif (LLVM_USE_SANITIZER STREQUAL "Leaks")
      append_common_sanitizer_flags()
      append("-fsanitize=leak" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
    else()
      message(FATAL_ERROR "Unsupported value of LLVM_USE_SANITIZER: ${LLVM_USE_SANITIZER}")
    endif()
  elseif(MINGW)
    if (LLVM_USE_SANITIZER STREQUAL "Address")
      append_common_sanitizer_flags()
      append("-fsanitize=address" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
    elseif (LLVM_USE_SANITIZER STREQUAL "Undefined")
      append_common_sanitizer_flags()
      append("${LLVM_UBSAN_FLAGS}" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
    elseif (LLVM_USE_SANITIZER STREQUAL "Address;Undefined" OR
            LLVM_USE_SANITIZER STREQUAL "Undefined;Address")
      append_common_sanitizer_flags()
      append("-fsanitize=address" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
      append("${LLVM_UBSAN_FLAGS}" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
    else()
      message(FATAL_ERROR "This sanitizer not yet supported in a MinGW environment: ${LLVM_USE_SANITIZER}")
    endif()
  elseif(MSVC)
    if (LLVM_USE_SANITIZER STREQUAL "Address")
      append_common_sanitizer_flags()
      append("-fsanitize=address" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
    else()
      message(FATAL_ERROR "This sanitizer not yet supported in the MSVC environment: ${LLVM_USE_SANITIZER}")
    endif()
  else()
    message(FATAL_ERROR "LLVM_USE_SANITIZER is not supported on this platform.")
  endif()
  if (LLVM_USE_SANITIZER MATCHES "(Undefined;)?Address(;Undefined)?")
    add_flag_if_supported("-fsanitize-address-use-after-scope"
                          FSANITIZE_USE_AFTER_SCOPE_FLAG)
  endif()
  if (LLVM_USE_SANITIZE_COVERAGE)
    append("-fsanitize=fuzzer-no-link" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
  endif()
  if (LLVM_USE_SANITIZER MATCHES ".*Undefined.*")
    set(IGNORELIST_FILE "${CMAKE_SOURCE_DIR}/utils/sanitizers/ubsan_ignorelist.txt")
    if (EXISTS "${IGNORELIST_FILE}")
      # Use this option name version since -fsanitize-ignorelist is only
      # accepted with clang 13.0 or newer.
      append("-fsanitize-blacklist=${IGNORELIST_FILE}"
             CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
    endif()
  endif()
endif()

# Turn on -gsplit-dwarf if requested in debug builds.
if (LLVM_USE_SPLIT_DWARF AND
    ((uppercase_CMAKE_BUILD_TYPE STREQUAL "DEBUG") OR
     (uppercase_CMAKE_BUILD_TYPE STREQUAL "RELWITHDEBINFO")))
  # Limit to clang and gcc so far. Add compilers supporting this option.
  if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR
      CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_options(-gsplit-dwarf)
  include(LLVMCheckLinkerFlag)
  llvm_check_linker_flag(CXX "-Wl,--gdb-index" LINKER_SUPPORTS_GDB_INDEX)
  append_if(LINKER_SUPPORTS_GDB_INDEX "-Wl,--gdb-index"
    CMAKE_EXE_LINKER_FLAGS CMAKE_MODULE_LINKER_FLAGS CMAKE_SHARED_LINKER_FLAGS)
  endif()
endif()

add_definitions( -D__STDC_CONSTANT_MACROS )
add_definitions( -D__STDC_FORMAT_MACROS )
add_definitions( -D__STDC_LIMIT_MACROS )

# clang and gcc don't default-print colored diagnostics when invoked from Ninja.
if (UNIX AND
    CMAKE_GENERATOR STREQUAL "Ninja" AND
    (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR
     (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND
      NOT (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.9))))
  append("-fdiagnostics-color" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
endif()

# lld doesn't print colored diagnostics when invoked from Ninja
if (UNIX AND CMAKE_GENERATOR STREQUAL "Ninja")
  include(LLVMCheckLinkerFlag)
  llvm_check_linker_flag(CXX "-Wl,--color-diagnostics" LINKER_SUPPORTS_COLOR_DIAGNOSTICS)
  append_if(LINKER_SUPPORTS_COLOR_DIAGNOSTICS "-Wl,--color-diagnostics"
    CMAKE_EXE_LINKER_FLAGS CMAKE_MODULE_LINKER_FLAGS CMAKE_SHARED_LINKER_FLAGS)
endif()

# Add flags for add_dead_strip().
# FIXME: With MSVS, consider compiling with /Gy and linking with /OPT:REF?
# But MinSizeRel seems to add that automatically, so maybe disable these
# flags instead if LLVM_NO_DEAD_STRIP is set.
if(NOT CYGWIN AND NOT MSVC)
  if(NOT ${CMAKE_SYSTEM_NAME} MATCHES "Darwin" AND
     NOT uppercase_CMAKE_BUILD_TYPE STREQUAL "DEBUG")
    check_c_compiler_flag("-Werror -fno-function-sections" C_SUPPORTS_FNO_FUNCTION_SECTIONS)
    if (C_SUPPORTS_FNO_FUNCTION_SECTIONS)
      # Don't add -ffunction-sections if it can't be disabled with -fno-function-sections.
      # Doing so will break sanitizers.
      add_flag_if_supported("-ffunction-sections" FFUNCTION_SECTIONS)
    elseif (CMAKE_CXX_COMPILER_ID MATCHES "XL")
      append("-qfuncsect" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
    endif()
    add_flag_if_supported("-fdata-sections" FDATA_SECTIONS)
  endif()
elseif(MSVC)
  if( NOT uppercase_CMAKE_BUILD_TYPE STREQUAL "DEBUG" )
    append("/Gw" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
  endif()
endif()

if(MSVC)
  # Remove flags here, for exceptions and RTTI.
  # Each target property or source property should be responsible to control
  # them.
  # CL.EXE complains to override flags like "/GR /GR-".
  string(REGEX REPLACE "(^| ) */EH[-cs]+ *( |$)" "\\1 \\2" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  string(REGEX REPLACE "(^| ) */GR-? *( |$)" "\\1 \\2" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

# Provide public options to globally control RTTI and EH
option(LLVM_ENABLE_EH "Enable Exception handling" OFF)
option(LLVM_ENABLE_RTTI "Enable run time type information" OFF)
if(LLVM_ENABLE_EH AND NOT LLVM_ENABLE_RTTI)
  message(FATAL_ERROR "Exception handling requires RTTI. You must set LLVM_ENABLE_RTTI to ON")
endif()

option(LLVM_USE_NEWPM "Build LLVM using the experimental new pass manager" Off)
mark_as_advanced(LLVM_USE_NEWPM)
if (LLVM_USE_NEWPM)
  append("-fexperimental-new-pass-manager"
    CMAKE_CXX_FLAGS
    CMAKE_C_FLAGS
    CMAKE_EXE_LINKER_FLAGS
    CMAKE_SHARED_LINKER_FLAGS)
endif()

option(LLVM_ENABLE_IR_PGO "Build LLVM and tools with IR PGO instrumentation (deprecated)" Off)
mark_as_advanced(LLVM_ENABLE_IR_PGO)

set(LLVM_BUILD_INSTRUMENTED OFF CACHE STRING "Build LLVM and tools with PGO instrumentation. May be specified as IR or Frontend")
set(LLVM_VP_COUNTERS_PER_SITE "1.5" CACHE STRING "Value profile counters to use per site for IR PGO with Clang")
mark_as_advanced(LLVM_BUILD_INSTRUMENTED LLVM_VP_COUNTERS_PER_SITE)
string(TOUPPER "${LLVM_BUILD_INSTRUMENTED}" uppercase_LLVM_BUILD_INSTRUMENTED)

if (LLVM_BUILD_INSTRUMENTED)
  if (LLVM_ENABLE_IR_PGO OR uppercase_LLVM_BUILD_INSTRUMENTED STREQUAL "IR")
    append("-fprofile-generate=\"${LLVM_PROFILE_DATA_DIR}\""
      CMAKE_CXX_FLAGS
      CMAKE_C_FLAGS)
    if(NOT LINKER_IS_LLD_LINK)
        append("-fprofile-generate=\"${LLVM_PROFILE_DATA_DIR}\""
          CMAKE_EXE_LINKER_FLAGS
          CMAKE_SHARED_LINKER_FLAGS)
    endif()
    # Set this to avoid running out of the value profile node section
    # under clang in dynamic linking mode.
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND
        CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 11 AND
        LLVM_LINK_LLVM_DYLIB)
      append("-Xclang -mllvm -Xclang -vp-counters-per-site=${LLVM_VP_COUNTERS_PER_SITE}"
        CMAKE_CXX_FLAGS
        CMAKE_C_FLAGS)
    endif()
  elseif(uppercase_LLVM_BUILD_INSTRUMENTED STREQUAL "CSIR")
    append("-fcs-profile-generate=\"${LLVM_CSPROFILE_DATA_DIR}\""
      CMAKE_CXX_FLAGS
      CMAKE_C_FLAGS)
    if(NOT LINKER_IS_LLD_LINK)
      append("-fcs-profile-generate=\"${LLVM_CSPROFILE_DATA_DIR}\""
        CMAKE_EXE_LINKER_FLAGS
        CMAKE_SHARED_LINKER_FLAGS)
    endif()
  else()
    append("-fprofile-instr-generate=\"${LLVM_PROFILE_FILE_PATTERN}\""
      CMAKE_CXX_FLAGS
      CMAKE_C_FLAGS)
    if(NOT LINKER_IS_LLD_LINK)
      append("-fprofile-instr-generate=\"${LLVM_PROFILE_FILE_PATTERN}\""
        CMAKE_EXE_LINKER_FLAGS
        CMAKE_SHARED_LINKER_FLAGS)
    endif()
  endif()
endif()

# When using clang-cl with an instrumentation-based tool, add clang's library
# resource directory to the library search path. Because cmake invokes the
# linker directly, it isn't sufficient to pass -fsanitize=* to the linker.
if (CLANG_CL AND (LLVM_BUILD_INSTRUMENTED OR LLVM_USE_SANITIZER))
  execute_process(
    COMMAND ${CMAKE_CXX_COMPILER} /clang:-print-libgcc-file-name /clang:--rtlib=compiler-rt
    OUTPUT_VARIABLE clang_compiler_rt_file
    ERROR_VARIABLE clang_cl_stderr
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE clang_cl_exit_code)
  if (NOT "${clang_cl_exit_code}" STREQUAL "0")
    message(FATAL_ERROR
      "Unable to invoke clang-cl to find resource dir: ${clang_cl_stderr}")
  endif()
  file(TO_CMAKE_PATH "${clang_compiler_rt_file}" clang_compiler_rt_file)
  get_filename_component(clang_runtime_dir "${clang_compiler_rt_file}" DIRECTORY)
  append("/libpath:${clang_runtime_dir}"
    CMAKE_EXE_LINKER_FLAGS
    CMAKE_MODULE_LINKER_FLAGS
    CMAKE_SHARED_LINKER_FLAGS)
endif()

if(LLVM_PROFDATA_FILE AND EXISTS ${LLVM_PROFDATA_FILE})
  if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" )
    append("-fprofile-instr-use=\"${LLVM_PROFDATA_FILE}\""
      CMAKE_CXX_FLAGS
      CMAKE_C_FLAGS)
    if(NOT LINKER_IS_LLD_LINK)
      append("-fprofile-instr-use=\"${LLVM_PROFDATA_FILE}\""
        CMAKE_EXE_LINKER_FLAGS
        CMAKE_SHARED_LINKER_FLAGS)
    endif()
  else()
    message(FATAL_ERROR "LLVM_PROFDATA_FILE can only be specified when compiling with clang")
  endif()
endif()

option(LLVM_BUILD_INSTRUMENTED_COVERAGE "Build LLVM and tools with Code Coverage instrumentation" Off)
mark_as_advanced(LLVM_BUILD_INSTRUMENTED_COVERAGE)
append_if(LLVM_BUILD_INSTRUMENTED_COVERAGE "-fprofile-instr-generate=\"${LLVM_PROFILE_FILE_PATTERN}\" -fcoverage-mapping"
  CMAKE_CXX_FLAGS
  CMAKE_C_FLAGS
  CMAKE_EXE_LINKER_FLAGS
  CMAKE_SHARED_LINKER_FLAGS)

if (LLVM_BUILD_INSTRUMENTED AND LLVM_BUILD_INSTRUMENTED_COVERAGE)
  message(FATAL_ERROR "LLVM_BUILD_INSTRUMENTED and LLVM_BUILD_INSTRUMENTED_COVERAGE cannot both be specified")
endif()

if(LLVM_ENABLE_LTO AND LLVM_ON_WIN32 AND NOT LINKER_IS_LLD_LINK AND NOT MINGW)
  message(FATAL_ERROR "When compiling for Windows, LLVM_ENABLE_LTO requires using lld as the linker (point CMAKE_LINKER at lld-link.exe)")
endif()
if(uppercase_LLVM_ENABLE_LTO STREQUAL "THIN")
  append("-flto=thin" CMAKE_CXX_FLAGS CMAKE_C_FLAGS)
  if(NOT LINKER_IS_LLD_LINK)
    append("-flto=thin" CMAKE_EXE_LINKER_FLAGS CMAKE_SHARED_LINKER_FLAGS)
  endif()
  # If the linker supports it, enable the lto cache. This improves initial build
  # time a little since we re-link a lot of the same objects, and significantly
  # improves incremental build time.
  # FIXME: We should move all this logic into the clang driver.
  if(APPLE)
    append("-Wl,-cache_path_lto,${PROJECT_BINARY_DIR}/lto.cache"
           CMAKE_EXE_LINKER_FLAGS CMAKE_SHARED_LINKER_FLAGS)
  elseif((UNIX OR MINGW) AND LLVM_USE_LINKER STREQUAL "lld")
    append("-Wl,--thinlto-cache-dir=${PROJECT_BINARY_DIR}/lto.cache"
           CMAKE_EXE_LINKER_FLAGS CMAKE_SHARED_LINKER_FLAGS)
  elseif(LLVM_USE_LINKER STREQUAL "gold")
    append("-Wl,--plugin-opt,cache-dir=${PROJECT_BINARY_DIR}/lto.cache"
           CMAKE_EXE_LINKER_FLAGS CMAKE_SHARED_LINKER_FLAGS)
  elseif(LINKER_IS_LLD_LINK)
    append("/lldltocache:${PROJECT_BINARY_DIR}/lto.cache"
           CMAKE_EXE_LINKER_FLAGS CMAKE_SHARED_LINKER_FLAGS)
  endif()
elseif(uppercase_LLVM_ENABLE_LTO STREQUAL "FULL")
  append("-flto=full" CMAKE_CXX_FLAGS CMAKE_C_FLAGS)
  if(NOT LINKER_IS_LLD_LINK)
    append("-flto=full" CMAKE_EXE_LINKER_FLAGS CMAKE_SHARED_LINKER_FLAGS)
  endif()
elseif(LLVM_ENABLE_LTO)
  append("-flto" CMAKE_CXX_FLAGS CMAKE_C_FLAGS)
  if(NOT LINKER_IS_LLD_LINK)
    append("-flto" CMAKE_EXE_LINKER_FLAGS CMAKE_SHARED_LINKER_FLAGS)
  endif()
endif()

# Set an AIX default for LLVM_EXPORT_SYMBOLS_FOR_PLUGINS based on whether we are
# doing dynamic linking (see below).
set(LLVM_EXPORT_SYMBOLS_FOR_PLUGINS_AIX_default OFF)
if (NOT (BUILD_SHARED_LIBS OR LLVM_LINK_LLVM_DYLIB))
  set(LLVM_EXPORT_SYMBOLS_FOR_PLUGINS_AIX_default ON)
endif()

# This option makes utils/extract_symbols.py be used to determine the list of
# symbols to export from LLVM tools. This is necessary when on AIX or when using
# MSVC if you want to allow plugins. On AIX we don't show this option, and we
# enable it by default except when the LLVM libraries are set up for dynamic
# linking (due to incompatibility). With MSVC, note that the plugin has to
# explicitly link against (exactly one) tool so we can't unilaterally turn on
# LLVM_ENABLE_PLUGINS when it's enabled.
CMAKE_DEPENDENT_OPTION(LLVM_EXPORT_SYMBOLS_FOR_PLUGINS
       "Export symbols from LLVM tools so that plugins can import them" OFF
       "NOT ${CMAKE_SYSTEM_NAME} MATCHES AIX" ${LLVM_EXPORT_SYMBOLS_FOR_PLUGINS_AIX_default})
if(BUILD_SHARED_LIBS AND LLVM_EXPORT_SYMBOLS_FOR_PLUGINS)
  message(FATAL_ERROR "BUILD_SHARED_LIBS not compatible with LLVM_EXPORT_SYMBOLS_FOR_PLUGINS")
endif()
if(LLVM_LINK_LLVM_DYLIB AND LLVM_EXPORT_SYMBOLS_FOR_PLUGINS)
  message(FATAL_ERROR "LLVM_LINK_LLVM_DYLIB not compatible with LLVM_EXPORT_SYMBOLS_FOR_PLUGINS")
endif()

# By default we should enable LLVM_ENABLE_IDE only for multi-configuration
# generators. This option disables optional build system features that make IDEs
# less usable.
set(LLVM_ENABLE_IDE_default OFF)
if (CMAKE_CONFIGURATION_TYPES)
  set(LLVM_ENABLE_IDE_default ON)
endif()
option(LLVM_ENABLE_IDE
       "Disable optional build system features that cause problems for IDE generators"
       ${LLVM_ENABLE_IDE_default})
if (CMAKE_CONFIGURATION_TYPES AND NOT LLVM_ENABLE_IDE)
  message(WARNING "Disabling LLVM_ENABLE_IDE on multi-configuration generators is not recommended.")
endif()

function(get_compile_definitions)
  get_directory_property(top_dir_definitions DIRECTORY ${CMAKE_SOURCE_DIR} COMPILE_DEFINITIONS)
  foreach(definition ${top_dir_definitions})
    if(DEFINED result)
      string(APPEND result " -D${definition}")
    else()
      set(result "-D${definition}")
    endif()
  endforeach()
  set(LLVM_DEFINITIONS "${result}" PARENT_SCOPE)
endfunction()
get_compile_definitions()

option(LLVM_FORCE_ENABLE_STATS "Enable statistics collection for builds that wouldn't normally enable it" OFF)

check_symbol_exists(os_signpost_interval_begin "os/signpost.h" macos_signposts_available)
if(macos_signposts_available)
  check_cxx_source_compiles(
    "#include <os/signpost.h>
    int main() { os_signpost_interval_begin(nullptr, 0, \"\", \"\"); return 0; }"
    macos_signposts_usable)
  if(macos_signposts_usable)
    set(LLVM_ENABLE_SUPPORT_XCODE_SIGNPOSTS "WITH_ASSERTS" CACHE STRING
        "Enable support for Xcode signposts. Can be WITH_ASSERTS, FORCE_ON, FORCE_OFF")
    string(TOUPPER "${LLVM_ENABLE_SUPPORT_XCODE_SIGNPOSTS}"
                   uppercase_LLVM_ENABLE_SUPPORT_XCODE_SIGNPOSTS)
    if( uppercase_LLVM_ENABLE_SUPPORT_XCODE_SIGNPOSTS STREQUAL "WITH_ASSERTS" )
      if( LLVM_ENABLE_ASSERTIONS )
        set( LLVM_SUPPORT_XCODE_SIGNPOSTS 1 )
      endif()
    elseif( uppercase_LLVM_ENABLE_SUPPORT_XCODE_SIGNPOSTS STREQUAL "FORCE_ON" )
      set( LLVM_SUPPORT_XCODE_SIGNPOSTS 1 )
    elseif( uppercase_LLVM_ENABLE_SUPPORT_XCODE_SIGNPOSTS STREQUAL "FORCE_OFF" )
      # We don't need to do anything special to turn off signposts.
    elseif( NOT DEFINED LLVM_ENABLE_SUPPORT_XCODE_SIGNPOSTS )
      # Treat LLVM_ENABLE_SUPPORT_XCODE_SIGNPOSTS like "FORCE_OFF" when it has not been
      # defined.
    else()
      message(FATAL_ERROR "Unknown value for LLVM_ENABLE_SUPPORT_XCODE_SIGNPOSTS:"
                          " \"${LLVM_ENABLE_SUPPORT_XCODE_SIGNPOSTS}\"!")
    endif()
  endif()
endif()

set(LLVM_SOURCE_PREFIX "" CACHE STRING "Use prefix for sources")

option(LLVM_USE_RELATIVE_PATHS_IN_DEBUG_INFO "Use relative paths in debug info" OFF)

if(LLVM_USE_RELATIVE_PATHS_IN_DEBUG_INFO)
  check_c_compiler_flag("-fdebug-prefix-map=foo=bar" SUPPORTS_FDEBUG_PREFIX_MAP)
  if(LLVM_ENABLE_PROJECTS_USED)
    get_filename_component(source_root "${LLVM_MAIN_SRC_DIR}/.." ABSOLUTE)
  else()
    set(source_root "${LLVM_MAIN_SRC_DIR}")
  endif()
  file(RELATIVE_PATH relative_root "${source_root}" "${CMAKE_BINARY_DIR}")
  append_if(SUPPORTS_FDEBUG_PREFIX_MAP "-fdebug-prefix-map=${CMAKE_BINARY_DIR}=${relative_root}" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
  append_if(SUPPORTS_FDEBUG_PREFIX_MAP "-fdebug-prefix-map=${source_root}/=${LLVM_SOURCE_PREFIX}" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
  add_flag_if_supported("-no-canonical-prefixes" NO_CANONICAL_PREFIXES)
endif()

option(LLVM_USE_RELATIVE_PATHS_IN_FILES "Use relative paths in sources and debug info" OFF)

if(LLVM_USE_RELATIVE_PATHS_IN_FILES)
  check_c_compiler_flag("-ffile-prefix-map=foo=bar" SUPPORTS_FFILE_PREFIX_MAP)
  if(LLVM_ENABLE_PROJECTS_USED)
    get_filename_component(source_root "${LLVM_MAIN_SRC_DIR}/.." ABSOLUTE)
  else()
    set(source_root "${LLVM_MAIN_SRC_DIR}")
  endif()
  file(RELATIVE_PATH relative_root "${source_root}" "${CMAKE_BINARY_DIR}")
  append_if(SUPPORTS_FFILE_PREFIX_MAP "-ffile-prefix-map=${CMAKE_BINARY_DIR}=${relative_root}" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
  append_if(SUPPORTS_FFILE_PREFIX_MAP "-ffile-prefix-map=${source_root}/=${LLVM_SOURCE_PREFIX}" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
  add_flag_if_supported("-no-canonical-prefixes" NO_CANONICAL_PREFIXES)
endif()

if(LLVM_INCLUDE_TESTS)
  # Lit test suite requires at least python 3.6
  set(LLVM_MINIMUM_PYTHON_VERSION 3.6)
else()
  # FIXME: it is unknown if this is the actual minimum bound
  set(LLVM_MINIMUM_PYTHON_VERSION 3.0)
endif()
