# This CMake module is responsible for interpreting the user defined LLVM_
# options and executing the appropriate CMake commands to realize the users'
# selections.

include(HandleLLVMStdlib)
include(AddLLVMDefinitions)
include(CheckCCompilerFlag)
include(CheckCXXCompilerFlag)

if(NOT LLVM_FORCE_USE_OLD_TOOLCHAIN)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.7)
      message(FATAL_ERROR "Host GCC version must be at least 4.7!")
    endif()
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 3.1)
      message(FATAL_ERROR "Host Clang version must be at least 3.1!")
    endif()

    # Also test that we aren't using too old of a version of libstdc++ with the
    # Clang compiler. This is tricky as there is no real way to check the
    # version of libstdc++ directly. Instead we test for a known bug in
    # libstdc++4.6 that is fixed in libstdc++4.7.
    if(NOT LLVM_ENABLE_LIBCXX)
      set(OLD_CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS})
      set(OLD_CMAKE_REQUIRED_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES})
      set(CMAKE_REQUIRED_FLAGS "-std=c++0x")
      if (ANDROID)
        set(CMAKE_REQUIRED_LIBRARIES "atomic")
      endif()
      check_cxx_source_compiles("
#include <atomic>
std::atomic<float> x(0.0f);
int main() { return (float)x; }"
        LLVM_NO_OLD_LIBSTDCXX)
      if(NOT LLVM_NO_OLD_LIBSTDCXX)
        message(FATAL_ERROR "Host Clang must be able to find libstdc++4.7 or newer!")
      endif()
      set(CMAKE_REQUIRED_FLAGS ${OLD_CMAKE_REQUIRED_FLAGS})
      set(CMAKE_REQUIRED_LIBRARIES ${OLD_CMAKE_REQUIRED_LIBRARIES})
    endif()
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 17.0)
      message(FATAL_ERROR "Host Visual Studio must be at least 2012 (MSVC 17.0)")
    endif()
  endif()
endif()

if( LLVM_ENABLE_ASSERTIONS )
  # MSVC doesn't like _DEBUG on release builds. See PR 4379.
  if( NOT MSVC )
    add_definitions( -D_DEBUG )
  endif()
  # On non-Debug builds cmake automatically defines NDEBUG, so we
  # explicitly undefine it:
  if( NOT uppercase_CMAKE_BUILD_TYPE STREQUAL "DEBUG" )
    add_definitions( -UNDEBUG )
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

if(WIN32)
  set(LLVM_HAVE_LINK_VERSION_SCRIPT 0)
  if(CYGWIN)
    set(LLVM_ON_WIN32 0)
    set(LLVM_ON_UNIX 1)
  else(CYGWIN)
    set(LLVM_ON_WIN32 1)
    set(LLVM_ON_UNIX 0)
  endif(CYGWIN)
  # Maximum path length is 160 for non-unicode paths
  set(MAXPATHLEN 160)
else(WIN32)
  if(UNIX)
    set(LLVM_ON_WIN32 0)
    set(LLVM_ON_UNIX 1)
    if(APPLE)
      set(LLVM_HAVE_LINK_VERSION_SCRIPT 0)
    else(APPLE)
      set(LLVM_HAVE_LINK_VERSION_SCRIPT 1)
    endif(APPLE)
    # FIXME: Maximum path length is currently set to 'safe' fixed value
    set(MAXPATHLEN 2024)
  else(UNIX)
    MESSAGE(SEND_ERROR "Unable to determine platform")
  endif(UNIX)
endif(WIN32)

set(EXEEXT ${CMAKE_EXECUTABLE_SUFFIX})
set(LTDL_SHLIB_EXT ${CMAKE_SHARED_LIBRARY_SUFFIX})

# We use *.dylib rather than *.so on darwin.
set(LLVM_PLUGIN_EXT ${CMAKE_SHARED_LIBRARY_SUFFIX})

if(APPLE)
  # Darwin-specific linker flags for loadable modules.
  set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -Wl,-flat_namespace -Wl,-undefined -Wl,suppress")
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
  if ("C_SUPPORTS_${name}" AND "CXX_SUPPORTS_${name}")
    message(STATUS "Building with ${flag}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${flag}" PARENT_SCOPE)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${flag}" PARENT_SCOPE)
  else()
    message(WARNING "${flag} is not supported.")
  endif()
endfunction()

if( LLVM_ENABLE_PIC )
  if( XCODE )
    # Xcode has -mdynamic-no-pic on by default, which overrides -fPIC. I don't
    # know how to disable this, so just force ENABLE_PIC off for now.
    message(WARNING "-fPIC not supported with Xcode.")
  elseif( WIN32 OR CYGWIN)
    # On Windows all code is PIC. MinGW warns if -fPIC is used.
  else()
    add_flag_or_print_warning("-fPIC" FPIC)

    if( WIN32 OR CYGWIN)
      # MinGW warns if -fvisibility-inlines-hidden is used.
    else()
      check_cxx_compiler_flag("-fvisibility-inlines-hidden" SUPPORTS_FVISIBILITY_INLINES_HIDDEN_FLAG)
      append_if(SUPPORTS_FVISIBILITY_INLINES_HIDDEN_FLAG "-fvisibility-inlines-hidden" CMAKE_CXX_FLAGS)
    endif()
  endif()
endif()

if( CMAKE_SIZEOF_VOID_P EQUAL 8 AND NOT WIN32 )
  # TODO: support other platforms and toolchains.
  if( LLVM_BUILD_32_BITS )
    message(STATUS "Building 32 bits executables and libraries.")
    add_llvm_definitions( -m32 )
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -m32")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -m32")
    set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -m32")
  endif( LLVM_BUILD_32_BITS )
endif( CMAKE_SIZEOF_VOID_P EQUAL 8 AND NOT WIN32 )

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
      add_llvm_definitions( /MP )
    else()
      message(STATUS "Number of parallel compiler jobs set to " ${LLVM_COMPILER_JOBS})
      add_llvm_definitions( /MP${LLVM_COMPILER_JOBS} )
    endif()
  else()
    message(STATUS "Parallel compilation disabled")
  endif()
endif()

if( MSVC )
  include(ChooseMSVCCRT)

  if( NOT (${CMAKE_VERSION} VERSION_LESS 2.8.11) )
    # set stack reserved size to ~10MB
    # CMake previously automatically set this value for MSVC builds, but the
    # behavior was changed in CMake 2.8.11 (Issue 12437) to use the MSVC default
    # value (1 MB) which is not enough for us in tasks such as parsing recursive
    # C++ templates in Clang.
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /STACK:10000000")
  endif()

  if( MSVC11 )
    add_llvm_definitions(-D_VARIADIC_MAX=10)
  endif()
  
  # Add definitions that make MSVC much less annoying.
  add_llvm_definitions(
    # For some reason MS wants to deprecate a bunch of standard functions...
    -D_CRT_SECURE_NO_DEPRECATE
    -D_CRT_SECURE_NO_WARNINGS
    -D_CRT_NONSTDC_NO_DEPRECATE
    -D_CRT_NONSTDC_NO_WARNINGS
    -D_SCL_SECURE_NO_DEPRECATE
    -D_SCL_SECURE_NO_WARNINGS

    # Disabled warnings.
    -wd4146 # Suppress 'unary minus operator applied to unsigned type, result still unsigned'
    -wd4180 # Suppress 'qualifier applied to function type has no meaning; ignored'
    -wd4244 # Suppress ''argument' : conversion from 'type1' to 'type2', possible loss of data'
    -wd4267 # Suppress ''var' : conversion from 'size_t' to 'type', possible loss of data'
    -wd4291 # Suppress ''declaration' : no matching operator delete found; memory will not be freed if initialization throws an exception'
    -wd4345 # Suppress 'behavior change: an object of POD type constructed with an initializer of the form () will be default-initialized'
    -wd4351 # Suppress 'new behavior: elements of array 'array' will be default initialized'
    -wd4355 # Suppress ''this' : used in base member initializer list'
    -wd4503 # Suppress ''identifier' : decorated name length exceeded, name was truncated'
    -wd4624 # Suppress ''derived class' : destructor could not be generated because a base class destructor is inaccessible'
    -wd4722 # Suppress 'function' : destructor never returns, potential memory leak
    -wd4800 # Suppress ''type' : forcing value to bool 'true' or 'false' (performance warning)'
    
    # Promoted warnings.
    -w14062 # Promote 'enumerator in switch of enum is not handled' to level 1 warning.

    # Promoted warnings to errors.
    -we4238 # Promote 'nonstandard extension used : class rvalue used as lvalue' to error.
    )

  # Enable warnings
  if (LLVM_ENABLE_WARNINGS)
    add_llvm_definitions( /W4 )
    if (LLVM_ENABLE_PEDANTIC)
      # No MSVC equivalent available
    endif (LLVM_ENABLE_PEDANTIC)
  endif (LLVM_ENABLE_WARNINGS)
  if (LLVM_ENABLE_WERROR)
    add_llvm_definitions( /WX )
  endif (LLVM_ENABLE_WERROR)
elseif( LLVM_COMPILER_IS_GCC_COMPATIBLE )
  if (LLVM_ENABLE_WARNINGS)
    append("-Wall -W -Wno-unused-parameter -Wwrite-strings" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)

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

    append_if(LLVM_ENABLE_PEDANTIC "-pedantic -Wno-long-long" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
    add_flag_if_supported("-Wcovered-switch-default" COVERED_SWITCH_DEFAULT_FLAG)
    append_if(USE_NO_UNINITIALIZED "-Wno-uninitialized" CMAKE_CXX_FLAGS)
    append_if(USE_NO_MAYBE_UNINITIALIZED "-Wno-maybe-uninitialized" CMAKE_CXX_FLAGS)
    check_cxx_compiler_flag("-Werror -Wnon-virtual-dtor" CXX_SUPPORTS_NON_VIRTUAL_DTOR_FLAG)
    append_if(CXX_SUPPORTS_NON_VIRTUAL_DTOR_FLAG "-Wnon-virtual-dtor" CMAKE_CXX_FLAGS)

    # Check if -Wcomment is OK with an // comment ending with '\' if the next
    # line is also a // comment.
    set(OLD_CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS})
    set(CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS} -Werror -Wcomment)
    CHECK_C_SOURCE_COMPILES("// \\\\\\n//\\nint main() {return 0;}"
                            C_WCOMMENT_ALLOWS_LINE_WRAP)
    set(CMAKE_REQUIRED_FLAGS ${OLD_CMAKE_REQUIRED_FLAGS})
    if (NOT C_WCOMMENT_ALLOWS_LINE_WRAP)
      append("-Wno-comment" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
    endif()
  endif (LLVM_ENABLE_WARNINGS)
  append_if(LLVM_ENABLE_WERROR "-Werror" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
  if (NOT LLVM_ENABLE_TIMESTAMPS)
    add_flag_if_supported("-Werror=date-time" WERROR_DATE_TIME)
  endif ()
  if (LLVM_ENABLE_CXX1Y)
    check_cxx_compiler_flag("-std=c++1y" CXX_SUPPORTS_CXX1Y)
    append_if(CXX_SUPPORTS_CXX1Y "-std=c++1y" CMAKE_CXX_FLAGS)
  else()
    check_cxx_compiler_flag("-std=c++11" CXX_SUPPORTS_CXX11)
    if (CXX_SUPPORTS_CXX11)
      if (CYGWIN OR MINGW)
        # MinGW and Cygwin are a bit stricter and lack things like
        # 'strdup', 'stricmp', etc in c++11 mode.
        append("-std=gnu++11" CMAKE_CXX_FLAGS)
      else()
        append("-std=c++11" CMAKE_CXX_FLAGS)
      endif()
    else()
      message(FATAL_ERROR "LLVM requires C++11 support but the '-std=c++11' flag isn't supported.")
    endif()
  endif()
endif( MSVC )

macro(append_common_sanitizer_flags)
  # Append -fno-omit-frame-pointer and turn on debug info to get better
  # stack traces.
  add_flag_if_supported("-fno-omit-frame-pointer" FNO_OMIT_FRAME_POINTER)
  if (NOT uppercase_CMAKE_BUILD_TYPE STREQUAL "DEBUG" AND
      NOT uppercase_CMAKE_BUILD_TYPE STREQUAL "RELWITHDEBINFO")
    add_flag_if_supported("-gline-tables-only" GLINE_TABLES_ONLY)
  endif()
  # Use -O1 even in debug mode, otherwise sanitizers slowdown is too large.
  if (uppercase_CMAKE_BUILD_TYPE STREQUAL "DEBUG")
    add_flag_if_supported("-O1" O1)
  endif()
endmacro()

# Turn on sanitizers if necessary.
if(LLVM_USE_SANITIZER)
  if (LLVM_ON_UNIX)
    if (LLVM_USE_SANITIZER STREQUAL "Address")
      append_common_sanitizer_flags()
      append("-fsanitize=address" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
    elseif (LLVM_USE_SANITIZER MATCHES "Memory(WithOrigins)?")
      append_common_sanitizer_flags()
      append("-fsanitize=memory" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
      if(LLVM_USE_SANITIZER STREQUAL "MemoryWithOrigins")
        append("-fsanitize-memory-track-origins" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
      endif()
    else()
      message(WARNING "Unsupported value of LLVM_USE_SANITIZER: ${LLVM_USE_SANITIZER}")
    endif()
  else()
    message(WARNING "LLVM_USE_SANITIZER is not supported on this platform.")
  endif()
endif()

# Turn on -gsplit-dwarf if requested
if(LLVM_USE_SPLIT_DWARF)
  add_llvm_definitions("-gsplit-dwarf")
endif()

add_llvm_definitions( -D__STDC_CONSTANT_MACROS )
add_llvm_definitions( -D__STDC_FORMAT_MACROS )
add_llvm_definitions( -D__STDC_LIMIT_MACROS )

# clang doesn't print colored diagnostics when invoked from Ninja
if (UNIX AND
    CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND
    CMAKE_GENERATOR STREQUAL "Ninja")
  append("-fcolor-diagnostics" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
endif()

# Add flags for add_dead_strip().
# FIXME: With MSVS, consider compiling with /Gy and linking with /OPT:REF?
# But MinSizeRel seems to add that automatically, so maybe disable these
# flags instead if LLVM_NO_DEAD_STRIP is set.
if(NOT CYGWIN AND NOT WIN32)
  if(NOT ${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    check_c_compiler_flag("-Werror -fno-function-sections" C_SUPPORTS_FNO_FUNCTION_SECTIONS)
    if (C_SUPPORTS_FNO_FUNCTION_SECTIONS)
      # Don't add -ffunction-section if it can be disabled with -fno-function-sections.
      # Doing so will break sanitizers.
      add_flag_if_supported("-ffunction-sections" FFUNCTION_SECTIONS)
    endif()
    add_flag_if_supported("-fdata-sections" FDATA_SECTIONS)
  endif()
endif()

if(CYGWIN OR MINGW)
  # Prune --out-implib from executables. It doesn't make sense even
  # with --export-all-symbols.
  string(REGEX REPLACE "-Wl,--out-implib,[^ ]+ " " "
    CMAKE_C_LINK_EXECUTABLE "${CMAKE_C_LINK_EXECUTABLE}")
  string(REGEX REPLACE "-Wl,--out-implib,[^ ]+ " " "
    CMAKE_CXX_LINK_EXECUTABLE "${CMAKE_CXX_LINK_EXECUTABLE}")
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

# Plugin support
# FIXME: Make this configurable.
if(WIN32 OR CYGWIN)
  if(BUILD_SHARED_LIBS)
    set(LLVM_ENABLE_PLUGINS ON)
  else()
    set(LLVM_ENABLE_PLUGINS OFF)
  endif()
else()
  set(LLVM_ENABLE_PLUGINS ON)
endif()
