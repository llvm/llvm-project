include(AddLLVMDefinitions)

# Run-time build mode; It is used for unittests.
if(MSVC_IDE)
  # Expect "$(Configuration)", "$(OutDir)", etc.
  # It is expanded by msbuild or similar.
  set(RUNTIME_BUILD_MODE "${CMAKE_CFG_INTDIR}")
elseif(NOT CMAKE_BUILD_TYPE STREQUAL "")
  # Expect "Release" "Debug", etc.
  # Or unittests could not run.
  set(RUNTIME_BUILD_MODE ${CMAKE_BUILD_TYPE})
else()
  # It might be "."
  set(RUNTIME_BUILD_MODE "${CMAKE_CFG_INTDIR}")
endif()

set(LIT_ARGS_DEFAULT "-sv")
if (MSVC OR XCODE)
  set(LIT_ARGS_DEFAULT "${LIT_ARGS_DEFAULT} --no-progress-bar")
endif()
set(LLVM_LIT_ARGS "${LIT_ARGS_DEFAULT}"
    CACHE STRING "Default options for lit")

if( LLVM_ENABLE_ASSERTIONS )
  # MSVC doesn't like _DEBUG on release builds. See PR 4379.
  if( NOT MSVC )
    add_definitions( -D_DEBUG )
  endif()
  # On Release builds cmake automatically defines NDEBUG, so we
  # explicitly undefine it:
  if( uppercase_CMAKE_BUILD_TYPE STREQUAL "RELEASE" )
    add_definitions( -UNDEBUG )
  endif()
else()
  if( NOT uppercase_CMAKE_BUILD_TYPE STREQUAL "RELEASE" )
    if( NOT MSVC_IDE AND NOT XCODE )
      add_definitions( -DNDEBUG )
    endif()
  endif()
endif()

if(WIN32)
  if(CYGWIN)
    set(LLVM_ON_WIN32 0)
    set(LLVM_ON_UNIX 1)
  else(CYGWIN)
    set(LLVM_ON_WIN32 1)
    set(LLVM_ON_UNIX 0)

    # This is effective only on Win32 hosts to use gnuwin32 tools.
    set(LLVM_LIT_TOOLS_DIR "" CACHE PATH "Path to GnuWin32 tools")
  endif(CYGWIN)
  set(LTDL_SHLIB_EXT ".dll")
  set(EXEEXT ".exe")
  # Maximum path length is 160 for non-unicode paths
  set(MAXPATHLEN 160)
else(WIN32)
  if(UNIX)
    set(LLVM_ON_WIN32 0)
    set(LLVM_ON_UNIX 1)
    if(APPLE)
      set(LTDL_SHLIB_EXT ".dylib")
    else(APPLE)
      set(LTDL_SHLIB_EXT ".so")
    endif(APPLE)
    set(EXEEXT "")
    # FIXME: Maximum path length is currently set to 'safe' fixed value
    set(MAXPATHLEN 2024)
  else(UNIX)
    MESSAGE(SEND_ERROR "Unable to determine platform")
  endif(UNIX)
endif(WIN32)

if( LLVM_ENABLE_PIC )
  if( XCODE )
    # Xcode has -mdynamic-no-pic on by default, which overrides -fPIC. I don't
    # know how to disable this, so just force ENABLE_PIC off for now.
    message(WARNING "-fPIC not supported with Xcode.")
  elseif( WIN32 )
    # On Windows all code is PIC. MinGW warns if -fPIC is used.
  else()
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-fPIC" SUPPORTS_FPIC_FLAG)
    if( SUPPORTS_FPIC_FLAG )
      message(STATUS "Building with -fPIC")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
      set(CMAKE_C_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
    else( SUPPORTS_FPIC_FLAG )
      message(WARNING "-fPIC not supported.")
    endif()
  endif()
endif()

if( CMAKE_SIZEOF_VOID_P EQUAL 8 AND NOT WIN32 )
  # TODO: support other platforms and toolchains.
  option(LLVM_BUILD_32_BITS "Build 32 bits executables and libraries." OFF)
  if( LLVM_BUILD_32_BITS )
    message(STATUS "Building 32 bits executables and libraries.")
    add_llvm_definitions( -m32 )
    list(APPEND CMAKE_EXE_LINKER_FLAGS -m32)
    list(APPEND CMAKE_SHARED_LINKER_FLAGS -m32)
  endif( LLVM_BUILD_32_BITS )
endif( CMAKE_SIZEOF_VOID_P EQUAL 8 AND NOT WIN32 )

if( MSVC_IDE AND ( MSVC90 OR MSVC10 ) )
  # Only Visual Studio 2008 and 2010 officially supports /MP.
  # Visual Studio 2005 do support it but it's experimental there.
  set(LLVM_COMPILER_JOBS "0" CACHE STRING
    "Number of parallel compiler jobs. 0 means use all processors. Default is 0.")
  if( NOT LLVM_COMPILER_JOBS STREQUAL "1" )
    if( LLVM_COMPILER_JOBS STREQUAL "0" )
      add_llvm_definitions( /MP )
    else()
      if (MSVC10)
        message(FATAL_ERROR
          "Due to a bug in CMake only 0 and 1 is supported for "
          "LLVM_COMPILER_JOBS when generating for Visual Studio 2010")
      else()
        message(STATUS "Number of parallel compiler jobs set to " ${LLVM_COMPILER_JOBS})
        add_llvm_definitions( /MP${LLVM_COMPILER_JOBS} )
      endif()
    endif()
  else()
    message(STATUS "Parallel compilation disabled")
  endif()
endif()

if( MSVC )
  include(ChooseMSVCCRT)

  # Add definitions that make MSVC much less annoying.
  add_llvm_definitions(
    # For some reason MS wants to deprecate a bunch of standard functions...
    -D_CRT_SECURE_NO_DEPRECATE
    -D_CRT_SECURE_NO_WARNINGS
    -D_CRT_NONSTDC_NO_DEPRECATE
    -D_CRT_NONSTDC_NO_WARNINGS
    -D_SCL_SECURE_NO_DEPRECATE
    -D_SCL_SECURE_NO_WARNINGS

    -wd4146 # Suppress 'unary minus operator applied to unsigned type, result still unsigned'
    -wd4180 # Suppress 'qualifier applied to function type has no meaning; ignored'
    -wd4224 # Suppress 'nonstandard extension used : formal parameter 'identifier' was previously defined as a type'
    -wd4244 # Suppress ''argument' : conversion from 'type1' to 'type2', possible loss of data'
    -wd4267 # Suppress ''var' : conversion from 'size_t' to 'type', possible loss of data'
    -wd4275 # Suppress 'An exported class was derived from a class that was not exported.'
    -wd4291 # Suppress ''declaration' : no matching operator delete found; memory will not be freed if initialization throws an exception'
    -wd4345 # Suppress 'behavior change: an object of POD type constructed with an initializer of the form () will be default-initialized'
    -wd4351 # Suppress 'new behavior: elements of array 'array' will be default initialized'
    -wd4355 # Suppress ''this' : used in base member initializer list'
    -wd4503 # Suppress ''identifier' : decorated name length exceeded, name was truncated'
    -wd4624 # Suppress ''derived class' : destructor could not be generated because a base class destructor is inaccessible'
    -wd4715 # Suppress ''function' : not all control paths return a value'
    -wd4800 # Suppress ''type' : forcing value to bool 'true' or 'false' (performance warning)'
    -wd4065 # Suppress 'switch statement contains 'default' but no 'case' labels'

    -w14062 # Promote "enumerator in switch of enum is not handled" to level 1 warning.
    )

  # Enable warnings
  if (LLVM_ENABLE_WARNINGS)
    add_llvm_definitions( /W4 /Wall )
    if (LLVM_ENABLE_PEDANTIC)
      # No MSVC equivalent available
    endif (LLVM_ENABLE_PEDANTIC)
  endif (LLVM_ENABLE_WARNINGS)
  if (LLVM_ENABLE_WERROR)
    add_llvm_definitions( /WX )
  endif (LLVM_ENABLE_WERROR)
elseif( CMAKE_COMPILER_IS_GNUCXX )
  if (LLVM_ENABLE_WARNINGS)
    add_llvm_definitions( -Wall -W -Wno-unused-parameter -Wwrite-strings )
    if (LLVM_ENABLE_PEDANTIC)
      add_llvm_definitions( -pedantic -Wno-long-long )
    endif (LLVM_ENABLE_PEDANTIC)
  endif (LLVM_ENABLE_WARNINGS)
  if (LLVM_ENABLE_WERROR)
    add_llvm_definitions( -Werror )
  endif (LLVM_ENABLE_WERROR)
endif( MSVC )

add_llvm_definitions( -D__STDC_LIMIT_MACROS )
add_llvm_definitions( -D__STDC_CONSTANT_MACROS )

option(LLVM_INCLUDE_TESTS "Generate build targets for the LLVM unit tests." ON)
