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

# CMAKE libomp
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

# Add cmake directory to search for custom cmake functions
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

# Standalone build or part of LLVM?
set(LIBOMP_STANDALONE_BUILD FALSE)
if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}" OR
   "${CMAKE_SOURCE_DIR}/runtime" STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}")
  project(libomp C CXX)
  set(LIBOMP_STANDALONE_BUILD TRUE)
endif()

# Set libomp version
set(LIBOMP_VERSION_MAJOR 5)
set(LIBOMP_VERSION_MINOR 0)

# These include files are in the cmake/ subdirectory
include(LibompUtils)
include(LibompGetArchitecture)
include(LibompHandleFlags)
include(LibompDefinitions)

# Determine the target architecture
if(${LIBOMP_STANDALONE_BUILD})
  # If adding a new architecture, take a look at cmake/LibompGetArchitecture.cmake
  libomp_get_architecture(LIBOMP_DETECTED_ARCH)
  set(LIBOMP_ARCH ${LIBOMP_DETECTED_ARCH} CACHE STRING
    "The architecture to build for (x86_64/i386/arm/ppc64/ppc64le/aarch64/mic).")
  # Allow user to choose a suffix for the installation directory.
  set(LIBOMP_LIBDIR_SUFFIX "" CACHE STRING
    "suffix of lib installation directory e.g., 64 => lib64")
  # Should assertions be enabled?  They are on by default.
  set(LIBOMP_ENABLE_ASSERTIONS TRUE CACHE BOOL
    "enable assertions?")
  set(LIBOMP_ENABLE_WERROR FALSE CACHE BOOL
    "Enable -Werror flags to turn warnings into errors for supporting compilers.")
  # CMAKE_BUILD_TYPE was not defined, set default to Release
  if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
  endif()
else() # Part of LLVM build
  # Determine the native architecture from LLVM.
  string(TOLOWER "${LLVM_TARGET_ARCH}" LIBOMP_NATIVE_ARCH)
  if( LIBOMP_NATIVE_ARCH STREQUAL "host" )
    string(REGEX MATCH "^[^-]*" LIBOMP_NATIVE_ARCH ${LLVM_HOST_TRIPLE})
  endif ()
  if(LIBOMP_NATIVE_ARCH MATCHES "i[2-6]86")
    set(LIBOMP_ARCH i386)
  elseif(LIBOMP_NATIVE_ARCH STREQUAL "x86")
    set(LIBOMP_ARCH i386)
  elseif(LIBOMP_NATIVE_ARCH STREQUAL "amd64")
    set(LIBOMP_ARCH x86_64)
  elseif(LIBOMP_NATIVE_ARCH STREQUAL "x86_64")
    set(LIBOMP_ARCH x86_64)
  elseif(LIBOMP_NATIVE_ARCH MATCHES "powerpc")
    set(LIBOMP_ARCH ppc64)
  elseif(LIBOMP_NATIVE_ARCH MATCHES "aarch64")
    set(LIBOMP_ARCH aarch64)
  elseif(LIBOMP_NATIVE_ARCH MATCHES "arm64")
    set(LIBOMP_ARCH aarch64)
  elseif(LIBOMP_NATIVE_ARCH MATCHES "arm")
    set(LIBOMP_ARCH arm)
  else()
    # last ditch effort
    libomp_get_architecture(LIBOMP_ARCH)
  endif ()
  set(LIBOMP_LIBDIR_SUFFIX ${LLVM_LIBDIR_SUFFIX})
  set(LIBOMP_ENABLE_ASSERTIONS ${LLVM_ENABLE_ASSERTIONS})
  set(LIBOMP_ENABLE_WERROR ${LLVM_ENABLE_WERROR})
endif()
libomp_check_variable(LIBOMP_ARCH 32e x86_64 32 i386 arm ppc64 ppc64le aarch64 mic)

set(LIBOMP_LIB_TYPE normal CACHE STRING
  "Performance,Profiling,Stubs library (normal/profile/stubs)")
libomp_check_variable(LIBOMP_LIB_TYPE normal profile stubs)
set(LIBOMP_OMP_VERSION 41 CACHE STRING
  "The OpenMP version (41/40/30)")
libomp_check_variable(LIBOMP_OMP_VERSION 41 40 30)
# Set the OpenMP Year and Month assiociated with version
if(${LIBOMP_OMP_VERSION} GREATER 40 OR ${LIBOMP_OMP_VERSION} EQUAL 40)
  set(LIBOMP_OMP_YEAR_MONTH 201307)
elseif(${LIBOMP_OMP_VERSION} GREATER 30 OR ${LIBOMP_OMP_VERSION} EQUAL 30)
  set(LIBOMP_OMP_YEAR_MONTH 201107)
else()
  set(LIBOMP_OMP_YEAR_MONTH 200505)
endif()
set(LIBOMP_MIC_ARCH knc CACHE STRING
  "Intel(R) Many Integrated Core Architecture (Intel(R) MIC Architecture) (knf/knc).  Ignored if not Intel(R) MIC Architecture build.")
if("${LIBOMP_ARCH}" STREQUAL "mic")
  libomp_check_variable(LIBOMP_MIC_ARCH knf knc)
endif()
set(LIBOMP_FORTRAN_MODULES FALSE CACHE BOOL
  "Create Fortran module files? (requires fortran compiler)")

# - Support for universal fat binary builds on Mac
# - Having this extra variable allows people to build this library as a universal library
#   without forcing a universal build of the llvm/clang compiler.
set(LIBOMP_OSX_ARCHITECTURES "${CMAKE_OSX_ARCHITECTURES}" CACHE STRING
  "For Mac builds, semicolon separated list of architectures to build for universal fat binary.")
set(CMAKE_OSX_ARCHITECTURES ${LIBOMP_OSX_ARCHITECTURES})

# User specified flags.  These are appended to the configured flags.
set(LIBOMP_CFLAGS "" CACHE STRING
  "Appended user specified C compiler flags.")
set(LIBOMP_CXXFLAGS "" CACHE STRING
  "Appended user specified C++ compiler flags.")
set(LIBOMP_CPPFLAGS "" CACHE STRING
  "Appended user specified C preprocessor flags.")
set(LIBOMP_ASMFLAGS "" CACHE STRING
  "Appended user specified assembler flags.")
set(LIBOMP_LDFLAGS "" CACHE STRING
  "Appended user specified linker flags.")
set(LIBOMP_LIBFLAGS "" CACHE STRING
  "Appended user specified linked libs flags. (e.g., -lm)")
set(LIBOMP_FFLAGS "" CACHE STRING
  "Appended user specified Fortran compiler flags.  These are only used if LIBOMP_FORTRAN_MODULES==TRUE.")

# Should the libomp library and generated headers be copied into the original source exports/ directory
# Turning this to FALSE aids parallel builds to not interfere with each other.
# Currently, the testsuite module expects the just built OpenMP library to be located inside the exports/
# directory.  TODO: have testsuite run under llvm-lit directly.  We can then get rid of copying to exports/
set(LIBOMP_COPY_EXPORTS TRUE CACHE STRING
  "Should exports be copied into source exports/ directory?")

# HWLOC-support
set(LIBOMP_USE_HWLOC FALSE CACHE BOOL
  "Use Hwloc (http://www.open-mpi.org/projects/hwloc/) library for affinity?")
set(LIBOMP_HWLOC_INSTALL_DIR /usr/local CACHE PATH
  "Install path for hwloc library")

# Get the build number from kmp_version.c
libomp_get_build_number("${CMAKE_CURRENT_SOURCE_DIR}" LIBOMP_VERSION_BUILD)
math(EXPR LIBOMP_VERSION_BUILD_YEAR "${LIBOMP_VERSION_BUILD}/10000")
math(EXPR LIBOMP_VERSION_BUILD_MONTH_DAY "${LIBOMP_VERSION_BUILD}%10000")

# Currently don't record any timestamps
set(LIBOMP_BUILD_DATE "No_Timestamp")

# Architecture
set(IA32 FALSE)
set(INTEL64 FALSE)
set(ARM FALSE)
set(AARCH64 FALSE)
set(PPC64BE FALSE)
set(PPC64LE FALSE)
set(PPC64 FALSE)
set(MIC FALSE)
if("${LIBOMP_ARCH}" STREQUAL "i386" OR "${LIBOMP_ARCH}" STREQUAL "32")    # IA-32 architecture
  set(IA32 TRUE)
elseif("${LIBOMP_ARCH}" STREQUAL "x86_64" OR "${LIBOMP_ARCH}" STREQUAL "32e") # Intel(R) 64 architecture
  set(INTEL64 TRUE)
elseif("${LIBOMP_ARCH}" STREQUAL "arm") # ARM architecture
  set(ARM TRUE)
elseif("${LIBOMP_ARCH}" STREQUAL "ppc64") # PPC64BE architecture
  set(PPC64BE TRUE)
  set(PPC64 TRUE)
elseif("${LIBOMP_ARCH}" STREQUAL "ppc64le") # PPC64LE architecture
  set(PPC64LE TRUE)
  set(PPC64 TRUE)
elseif("${LIBOMP_ARCH}" STREQUAL "aarch64") # AARCH64 architecture
  set(AARCH64 TRUE)
elseif("${LIBOMP_ARCH}" STREQUAL "mic") # Intel(R) Many Integrated Core Architecture
  set(MIC TRUE)
endif()

# Set some flags based on build_type
set(RELEASE_BUILD FALSE)
set(DEBUG_BUILD FALSE)
set(RELWITHDEBINFO_BUILD FALSE)
set(MINSIZEREL_BUILD FALSE)
string(TOLOWER "${CMAKE_BUILD_TYPE}" libomp_build_type_lowercase)
if("${libomp_build_type_lowercase}" STREQUAL "release")
  set(RELEASE_BUILD TRUE)
elseif("${libomp_build_type_lowercase}" STREQUAL "debug")
  set(DEBUG_BUILD TRUE)
elseif("${libomp_build_type_lowercase}" STREQUAL "relwithdebinfo")
  set(RELWITHDEBINFO_BUILD TRUE)
elseif("${libomp_build_type_lowercase}" STREQUAL "minsizerel")
  set(MINSIZEREL_BUILD TRUE)
endif()

# Include itt notify interface? Right now, always.
set(LIBOMP_USE_ITT_NOTIFY TRUE)

# normal, profile, stubs library.
set(NORMAL_LIBRARY FALSE)
set(STUBS_LIBRARY FALSE)
set(PROFILE_LIBRARY FALSE)
if("${LIBOMP_LIB_TYPE}" STREQUAL "normal")
  set(NORMAL_LIBRARY TRUE)
elseif("${LIBOMP_LIB_TYPE}" STREQUAL "profile")
  set(PROFILE_LIBRARY TRUE)
elseif("${LIBOMP_LIB_TYPE}" STREQUAL "stubs")
  set(STUBS_LIBRARY TRUE)
endif()

# Setting directory names
set(LIBOMP_BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(LIBOMP_SRC_DIR ${LIBOMP_BASE_DIR}/src)
set(LIBOMP_TOOLS_DIR ${LIBOMP_BASE_DIR}/tools)
set(LIBOMP_INC_DIR ${LIBOMP_SRC_DIR}/include/${LIBOMP_OMP_VERSION})
set(LIBOMP_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

# Enabling Fortran if it is needed
if(${LIBOMP_FORTRAN_MODULES})
  enable_language(Fortran)
endif()
# Enable MASM Compiler if it is needed (Windows only)
if(WIN32)
  enable_language(ASM_MASM)
endif()

# Getting legal type/arch
libomp_get_legal_type(LIBOMP_LEGAL_TYPE)
libomp_get_legal_arch(LIBOMP_LEGAL_ARCH)

# Compiler flag checks, library checks, threading check, etc.
include(config-ix)

# Is there a quad precision data type available?
# TODO: Make this a real feature check
set(LIBOMP_USE_QUAD_PRECISION "${LIBOMP_HAVE_QUAD_PRECISION}" CACHE BOOL
  "Should 128-bit precision entry points be built?")
if(LIBOMP_USE_QUAD_PRECISION AND (NOT LIBOMP_HAVE_QUAD_PRECISION))
  libomp_error_say("128-bit quad precision functionality requested but not available")
endif()

# libgomp drop-in compatibility requires versioned symbols
set(LIBOMP_USE_VERSION_SYMBOLS "${LIBOMP_HAVE_VERSION_SYMBOLS}" CACHE BOOL
  "Should version symbols be used? These provide binary compatibility with libgomp.")
if(LIBOMP_USE_VERSION_SYMBOLS AND (NOT LIBOMP_HAVE_VERSION_SYMBOLS))
  libomp_error_say("Version symbols functionality requested but not available")
endif()

# On multinode systems, larger alignment is desired to avoid false sharing
set(LIBOMP_USE_INTERNODE_ALIGNMENT FALSE CACHE BOOL
  "Should larger alignment (4096 bytes) be used for some locks and data structures?")

# Build code that allows the OpenMP library to conveniently interface with debuggers
set(LIBOMP_USE_DEBUGGER FALSE CACHE BOOL
  "Enable debugger interface code?")

# Should we link to C++ library?
set(LIBOMP_USE_STDCPPLIB FALSE CACHE BOOL
  "Should we link to C++ library?")

# TSX (x86) based locks have __asm code which can be troublesome for some compilers.
# TODO: Make this a real feature check
set(LIBOMP_USE_ADAPTIVE_LOCKS "${LIBOMP_HAVE_ADAPTIVE_LOCKS}" CACHE BOOL
  "Should TSX-based lock be compiled (adaptive lock in kmp_lock.cpp).  These are x86 specific.")
if(LIBOMP_USE_ADAPTIVE_LOCKS AND (NOT LIBOMP_HAVE_ADAPTIVE_LOCKS))
  libomp_error_say("Adaptive locks (TSX) functionality requested but not available")
endif()

# - stats-gathering enables OpenMP stats where things like the number of
# parallel regions, clock ticks spent in particular openmp regions are recorded.
set(LIBOMP_STATS FALSE CACHE BOOL
  "Stats-Gathering functionality?")
if(LIBOMP_STATS AND (NOT LIBOMP_HAVE_STATS))
  libomp_error_say("Stats-gathering functionality requested but not available")
endif()
# The stats functionality requires the std c++ library
if(LIBOMP_STATS)
  set(LIBOMP_USE_STDCPPLIB TRUE)
endif()

# Shared library can be switched to a static library
set(LIBOMP_ENABLE_SHARED TRUE CACHE BOOL
  "Shared library instead of static library?")

if(WIN32 AND NOT LIBOMP_ENABLE_SHARED)
  libomp_error_say("Static libraries requested but not available on Windows")
endif()

# OMPT-support
set(LIBOMP_OMPT_DEBUG FALSE CACHE BOOL
  "Trace OMPT initialization?")
set(LIBOMP_OMPT_SUPPORT FALSE CACHE BOOL
  "OMPT-support?")
set(LIBOMP_OMPT_BLAME TRUE CACHE BOOL
  "OMPT-blame?")
set(LIBOMP_OMPT_TRACE TRUE CACHE BOOL
  "OMPT-trace?")
if(LIBOMP_OMPT_SUPPORT AND (NOT LIBOMP_HAVE_OMPT_SUPPORT))
  libomp_error_say("OpenMP Tools Interface requested but not available")
endif()

# Error check hwloc support after config-ix has run
if(LIBOMP_USE_HWLOC AND (NOT LIBOMP_HAVE_HWLOC))
  libomp_error_say("Hwloc requested but not available")
endif()

# Setting final library name
set(LIBOMP_DEFAULT_LIB_NAME libomp)
if(${PROFILE_LIBRARY})
  set(LIBOMP_DEFAULT_LIB_NAME ${LIBOMP_DEFAULT_LIB_NAME}prof)
endif()
if(${STUBS_LIBRARY})
  set(LIBOMP_DEFAULT_LIB_NAME ${LIBOMP_DEFAULT_LIB_NAME}stubs)
endif()
set(LIBOMP_LIB_NAME ${LIBOMP_DEFAULT_LIB_NAME} CACHE STRING "Base OMP library name")

if(${LIBOMP_ENABLE_SHARED})
  set(LIBOMP_LIBRARY_SUFFIX ${CMAKE_SHARED_LIBRARY_SUFFIX})
  set(LIBOMP_LIBRARY_KIND SHARED) 
  set(LIBOMP_INSTALL_KIND LIBRARY) 
else()
  set(LIBOMP_LIBRARY_SUFFIX ${CMAKE_STATIC_LIBRARY_SUFFIX})
  set(LIBOMP_LIBRARY_KIND STATIC) 
  set(LIBOMP_INSTALL_KIND ARCHIVE) 
endif()

set(LIBOMP_LIB_FILE ${LIBOMP_LIB_NAME}${LIBOMP_LIBRARY_SUFFIX})

# Print configuration after all variables are set.
if(${LIBOMP_STANDALONE_BUILD})
  libomp_say("Operating System     -- ${CMAKE_SYSTEM_NAME}")
  libomp_say("Target Architecture  -- ${LIBOMP_ARCH}")
  if(${MIC})
    libomp_say("Intel(R) MIC Architecture    -- ${LIBOMP_MIC_ARCH}")
  endif()
  libomp_say("Build Type           -- ${CMAKE_BUILD_TYPE}")
  libomp_say("OpenMP Version       -- ${LIBOMP_OMP_VERSION}")
  libomp_say("Library Kind         -- ${LIBOMP_LIBRARY_KIND}")
  libomp_say("Library Type         -- ${LIBOMP_LIB_TYPE}")
  libomp_say("Fortran Modules      -- ${LIBOMP_FORTRAN_MODULES}")
  # will say development if all zeros
  if(${LIBOMP_VERSION_BUILD} STREQUAL 00000000)
    set(LIBOMP_BUILD Development)
  else()
    set(LIBOMP_BUILD ${LIBOMP_VERSION_BUILD})
  endif()
  libomp_say("Build                -- ${LIBOMP_BUILD}")
  libomp_say("Use Stats-gathering  -- ${LIBOMP_STATS}")
  libomp_say("Use Debugger-support -- ${LIBOMP_USE_DEBUGGER}")
  libomp_say("Use OMPT-support     -- ${LIBOMP_OMPT_SUPPORT}")
  if(${LIBOMP_OMPT_SUPPORT})
    libomp_say("Use OMPT-blame       -- ${LIBOMP_OMPT_BLAME}")
    libomp_say("Use OMPT-trace       -- ${LIBOMP_OMPT_TRACE}")
  endif()
  libomp_say("Use Adaptive locks   -- ${LIBOMP_USE_ADAPTIVE_LOCKS}")
  libomp_say("Use quad precision   -- ${LIBOMP_USE_QUAD_PRECISION}")
  libomp_say("Use Hwloc library    -- ${LIBOMP_USE_HWLOC}")
endif()

add_subdirectory(src)
add_subdirectory(test)

